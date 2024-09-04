import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import  transforms
import torch.multiprocessing as mp

import utils
from args import get_parse_args
from torchvision.datasets import ImageFolder
from model.vqsrs import Model_vqsrs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def train_vqsrs(rank,world_size,args):
    args.rank = rank
    args.world_size = world_size
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.ToTensor(),
        utils.MyRotateTransform([90,180,270]),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    val_dataset = ImageFolder(args.data_val_path, transform=val_transform)
    re_dataset = ImageFolder(args.re_path, transform=val_transform) # Folder of images to be reconstructed
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=12,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    re_data_loader = torch.utils.data.DataLoader(
        re_dataset,
        batch_size=7,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    print(f"Data_val loaded: there are {len(val_dataset)} images.")
    print(f"Data_re loaded: there are {len(re_dataset)} images.")

    for data,label in re_data_loader: 
        fixed_images=data
        break
    fixed_images = fixed_images[:7]

    # ============ building networks ... ============
    # move networks to gpu
    model = Model_vqsrs().cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
   
    model = nn.parallel.DistributedDataParallel(model, 
                                                device_ids=[args.rank],
                                                find_unused_parameters=True,
                                                )

    # ============ preparing loss ... ============
    vqsrs_loss = utils.Loss_base().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups) 
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        vqsrs_loss=vqsrs_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of VQSRS ... ============
        train_stats = train_one_epoch(model, vqsrs_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        # if args.rank == 0:
        reconstruction = utils.generate_samples(fixed_images, model, args)
        reconstruction = reconstruction[:7]
        utils.plot_re(fixed_images, reconstruction.cpu(), args, epoch)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'vqsrs_loss': vqsrs_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        print("Epoch finished.")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(model, vqsrs_loss, data_loader, 
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (images, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        # move images to gpu
        images = images.cuda()
        label = label.cuda()
        output,msel,fc1,fc2,vq1loss,vq2loss,_,_,_= model(images)

        mse = vqsrs_loss.mse(output,images)
        fc1loss = vqsrs_loss.criterion(fc1, label)
        fc2loss = vqsrs_loss.criterion(fc2, label)
        loss = mse + msel + fc1loss + fc2loss + vq1loss + vq2loss
        f1acc = utils.accu(fc1,label,topk=(1,))
        f2acc = utils.accu(fc2,label,topk=(1,))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(mse=mse.item())
        metric_logger.update(fc1loss=fc1loss.item())
        metric_logger.update(trf1acc=f1acc[0].item())       
        metric_logger.update(fc2loss=fc2loss.item())
        metric_logger.update(trf2acc=f2acc[0].item()) 
        metric_logger.update(vq1loss=vq1loss.item())
        metric_logger.update(vq2loss=vq2loss.item())
        metric_logger.update(msel=msel.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.homepath).mkdir(parents=True, exist_ok=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train_vqsrs,args=(world_size,args,),nprocs=world_size)
    
