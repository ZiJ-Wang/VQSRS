import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.eff2d import efficientenc_b0

default_block_args = [
    # block arguments for the first encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 1,
                'kernel': 3,
                'stride': 1,
                'input_channels': 32,
                'out_channels': 16,
                'num_layers': 1,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 16,
                'out_channels': 24,
                'num_layers': 2,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 1,
                'input_channels': 24,
                'out_channels': 40,
                'num_layers': 2,
            },
        ]
    },
    # block arguments for the second encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 40,
                'out_channels': 80,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,  # 1 in the original
                'input_channels': 80,
                'out_channels': 112,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,
                'input_channels': 112,
                'out_channels': 192,
                'num_layers': 4,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 1,
                'input_channels': 192,
                'out_channels': 320,
                'num_layers': 1,
            },
        ]
    },
]

class effi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 1
        self.encoder_args = default_block_args
        self.encoder_args[0].update(
            {
                'in_channels': 3,
                'out_channels': 64,
                'first_layer_stride':2,
            }
        )
        self.encoder1 = efficientenc_b0(**self.encoder_args[0])

        self.encoder_args[1].update(
            {
                'in_channels': 64,
                'out_channels': 320,#576
                'first_layer_stride':1,
            }
        )
        self.encoder2 = efficientenc_b0(**self.encoder_args[1])

    def forward(self, x):
        enb = self.encoder1(x)
        ent = self.encoder2(enb)
        return enb,ent

   
class VectorQuantizer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 initialization = 'uniform',

                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if initialization == 'uniform':
            self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self._commitment_cost = 0.25

    def forward(self, input):
        in_re = input.permute(0, 2, 3, 1).contiguous()
        input_shape = in_re.shape
        #flatten in_re
        x_fla = in_re.view(-1, self.embedding_dim)

        dist = (torch.sum(x_fla**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(x_fla, self.embedding.weight.t()))
        
        # Obtain the encoding indices for each image and perform hierarchical clustering using the Pearson correlation coefficient
        encoding_indices = torch.argmin(dist, dim=1)
        self.encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)

        # Count histogram
        index_histogram = torch.stack(
            list(
                map(
                    lambda x: torch.histc(x, bins=self.num_embeddings, min=0, max=self.num_embeddings - 1),
                    encoding_indices.view((input.shape[0], -1)).float(),
                )
            )
        )

        encoding = torch.zeros(self.encoding_indices.shape[0], self.num_embeddings, device=input.device)
        encoding.scatter_(1, self.encoding_indices, 1)

        #quantize and unflatten
        quantized = torch.matmul(encoding, self.embedding.weight).view(input_shape)

        #loss
        e_latent_loss = F.mse_loss(quantized.detach(), in_re)
        q_latent_loss = F.mse_loss(quantized, in_re.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = in_re + (quantized - in_re).detach()
        avg_probs = torch.mean(encoding, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized,loss,index_histogram


class Res(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),   #silu = swish
            nn.Conv2d(dim,dim,3,1,1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv1(x)


class decoder2(nn.Module):
    def __init__(self,in_channel,outchannel) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res1 = Res(64)
        self.res2 = Res(64)
        self.up1_2 = nn.ZeroPad2d((1,0,1,0))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res3 = Res(32)
        self.res4 = Res(32)
        self.up2_3 = nn.ZeroPad2d((1,0,1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32,3,1,),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res5 = Res(32)
        self.res6 = Res(32)
        self.up3_4 = nn.ZeroPad2d((1,0,1,0))
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.up1_2(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.up2_3(x)
        x = self.conv3(x)
        x = self.up3(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.up3_4(x)
        x = self.conv4(x)
        return x


class decoder1(nn.Module):
    def __init__(self,in_channel,outchannel) -> None:
        super().__init__()
        self.up_en = nn.UpsamplingBilinear2d(size=(75,75))
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.res0 = Res(128)
        self.res01 = Res(128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res1 = Res(64)
        self.res2 = Res(64)
        self.up1_2 = nn.ZeroPad2d((1,1,1,1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res3 = Res(32)
        self.res4 = Res(32)
        self.up2_3 = nn.ZeroPad2d((1,1,1,1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32,3,1,),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32,outchannel,3,1,1),
        )
        
    def forward(self, x,y):
        y = self.up_en(y)
        x = torch.cat([x,y],dim=1)
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res01(x)

        x = self.conv1(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.up1_2(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.up2_3(x)
        x = self.conv3(x)
        return x


class Fcblock(nn.Module):
    def __init__(self,inchannel) -> None:
        super().__init__()
        self.inch = inchannel
        self.fc1 = nn.Sequential(
            nn.Linear(self.inch,7),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class Model_vqsrs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = effi()
        self.vq2 = VectorQuantizer(64,256) #self.vq2 = VectorQuantizer(64,256)
        self.decoder2 = decoder2(320,64)
        self.fc2 = Fcblock(320*10*10)
        self.fc1 = Fcblock(64*75*75)
        self.decoder1 = decoder1(448,3)
        self.vq1 = VectorQuantizer(64,256) #self.vq1 = VectorQuantizer(64,256)

    def forward(self, x):
        enb,ent = self.encoder(x)
        vq2,vq2loss,indexs2= self.vq2(ent)
        
        vq2fla = torch.flatten(vq2,start_dim=1)
        f2 = self.fc2(vq2fla)
        de2 = self.decoder2(vq2)
        de2_padded = F.pad(de2, (1, 1, 1, 1))
        vq1_pre = torch.cat([enb,de2_padded],dim=1)
        vq1,vq1loss,indexs1 = self.vq1(vq1_pre)
        de1 = self.decoder1(vq1,vq2)
        vq1fla = torch.flatten(enb,start_dim=1)
        f1 = self.fc1(vq1fla)
        msel = F.mse_loss(de2_padded,enb)
        return de1,msel,f1,f2,vq1loss,vq2loss,vq2,indexs1,indexs2