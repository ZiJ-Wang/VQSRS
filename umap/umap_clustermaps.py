from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt
import os,sys
from tqdm import tqdm
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from score import ARI,NMI,F1,CH,SC,dunn_index
from umap_utils import umap2d,umap_init_savepath,selfpearson_multi
from args import get_parse_args
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model.vqsrs import Model_vqsrs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = get_parse_args()
args.model_path = "result/p/checkpoint0020.pth"
umap_init_savepath(args)
print(args.savepath_dict["umap_figures"])

model = Model_vqsrs()
model_dict = model.state_dict()
state = torch.load(args.model_path)
state_dict = state['model']
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith("module.") :
            # remove prefix
        # print(k)
        state_dict[k[len("module.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
pre = {k: v for k, v in state_dict.items() if k in model_dict}
#print(pre.keys()) 
model_dict.update(pre)
model.load_state_dict(model_dict)

batch_size = 8
test_dataset = ImageFolder("your_datasets/val",transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,
                                     num_workers = 16,prefetch_factor=32)
new_idx = {v:k for k,v in test_loader.dataset.class_to_idx.items()}
print(new_idx)

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

output, output_label, output_pre = [], [], []
label_pro = []
label_pred=[]
indexs = []
model = model.to(device)
model.eval()
acc = [0, 0, 0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0, 0, 0]

num_classes = 7
confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
predicted_scores = []
true_labels = []

for i, _batch in enumerate(tqdm(test_loader, desc='Infer')):
    timg = _batch[0].to(device)
    tlabel = _batch[1].to(device)

    with torch.no_grad():
        outp = model(timg)
    out = outp[6]

    pre_ = outp[3]
    pre = torch.argsort(pre_,dim=1,descending=True).cpu().detach().numpy()

    index = outp[8]
    indexs.append(index.cpu())

    if not torch.is_tensor(out):
        out = out[4]
    output.append(out.detach().cpu().numpy())

    label = np.reshape(np.array(_batch[1]), (-1, 1))

    for i in range(min(batch_size, len(_batch[1]))):
        label_ = _batch[1][i].cpu().detach().numpy()
        prediction = pre[i][0]

        output_pre.append(prediction)

        confusion_matrix[int(label_)][int(prediction)] += 1

        if prediction == label_:
            acc[label_] += 1
        else:
            error[label_] += 1

    output_label.append(label)


label_true=[]
output = np.vstack(output)
output_label = np.vstack(output_label)
output_pre = np.vstack(output_pre)
for i,da in enumerate(output_label):
    label_true.append(new_idx[output_label[i][0]])
label_true = np.array(label_true)

print('acc:',sum(acc)/(sum(acc)+sum(error)))
print("Confusion Matrix:")
print(confusion_matrix)


data = np.vstack(indexs)
all_zero_columns = np.all(data == 0, axis=0)
zero_columns_indices = np.where(all_zero_columns)[0]
data = np.delete(data, zero_columns_indices, axis=1)
unique_labels = np.unique(label_true)
cellid_by_idx = np.zeros((len(unique_labels), data.shape[-1]))
fig, axes = plt.subplots(7, 1, figsize=(8, 16))
colors = [(220/255, 185/255, 60/255),   
          (245/255, 140/255, 30/255),  
          (100/255, 190/255, 70/255),  
          (0/255, 145/255, 145/255),  
          (125/255, 210/255, 245/255),  
          (80/255, 140/255, 200/255),  
          (35/255, 65/255, 155/255)] 
for i, cid in enumerate(tqdm(unique_labels)):
    data0 = data[label_true == cid]
    cellid_by_idx[i, :] = data0.sum(0) / data0.shape[0]

corr_idx_idx = np.nan_to_num(selfpearson_multi(cellid_by_idx.T, num_workers=8))

print('computing clustermaps...')
heatmap = sns.clustermap(
    corr_idx_idx,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
heatmap.ax_col_dendrogram.set_title(f'vqvae indhist Pearson corr hierarchy link')
heatmap.ax_heatmap.set_xlabel('vq index')
heatmap.ax_heatmap.set_ylabel('vq index')

feature_spectrum_indices = np.array(heatmap.dendrogram_row.reordered_ind)

heatmap.savefig(args.homepath+"/analysis/feature_spectra_figures/heatmap.png", dpi=300)

print('finish clustermaps')

print('computing Feature spectrum...')
from matplotlib.ticker import MultipleLocator
for i, cid in enumerate(tqdm(unique_labels)):
    x = range(len(cellid_by_idx[i, :]))
    axes[i].bar(x, cellid_by_idx[i, :], width=1, color=colors[i], alpha=0.7)
    axes[i].set_xticks(x, feature_spectrum_indices[x])
    x_major_locator = MultipleLocator(10)
    axes[i].xaxis.set_major_locator(x_major_locator)
    axes[i].set_ylim(0, 10)

    axes[i].spines[['right', 'top']].set_visible(False) 
    axes[i].set_title(f'{cid}')
    axes[i].set_xlabel('Feature index')
    axes[i].set_ylabel('Value')

fig.tight_layout()
fig.savefig(join(args.homepath+'/analysis/feature_spectra_figures/Average feature spectrum.png'), dpi=300)


umap_data = umap2d(
                    args = args,
                    embed=output,
                    labels=label_true,
                    title = "UMAP",
                    xlabel = "UMAP1",
                    ylabel = "UMAP2",
                    s = 0.3,
                    alpha = 0.5,
                    show_legend = True,
                )

print('finish umap')

# Calculating the score
F1_score = F1(output_label.flatten(), output_pre.flatten())
print('F1 score:',F1_score)
NMI_score = NMI(output_label.flatten(), output_pre.flatten())
print('NMI score:',NMI_score)
ARI_score = ARI(output_label.flatten(), output_pre.flatten())
print('ARI score:',ARI_score)

CH_score = CH(umap_data,output_pre.flatten())
print('CH_score:',CH_score)
SC_score = SC(umap_data,output_pre.flatten())
print('SC_score:',SC_score)
DUNN_score = dunn_index(umap_data,output_pre.flatten())
print('DUNN_score:',DUNN_score)