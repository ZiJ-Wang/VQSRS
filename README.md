<h1 align="center">VQSRS</h1>

<div style="text-align: center;">
    <b>Fast Real‑Time Brain Tumor Detection Based on Stimulated Raman Histology and Self‑Supervised Deep Learning Model</b>
</div>

<div style="text-align: center;">
    Zijun Wang, Kaitai Han, Wu Liu, Zhenghui Wang, Chaojing Shi, Xi Liu, Mengyuan Huang, Guocheng Sun, Shitou Liu, Qianjin Guo*
</div>

<div style="text-align: center;">
    Paper Link (<a href="https://doi.org/10.1007/s10278-024-01001-4">doi: 10.1007/s10278-024-01001-4</a>)
</div>



------

## Abstract

In intraoperative brain cancer procedures, real-time diagnosis is essential for ensuring safe and effective care. The prevailing workflow, which relies on histological staining with hematoxylin and eosin (H&E) for tissue processing, is resource-intensive, time-consuming, and requires considerable labor. Recently, an innovative approach combining Stimulated Raman histology (SRH) and deep convolutional neural networks (CNN) has emerged, creating a new avenue for real-time cancer diagnosis during surgery. While this approach exhibits potential, there exists an opportunity for refinement in the domain of feature extraction. In this study, we employ Coherent Raman scattering imaging method and a self-supervised deep learning model (VQVAE2) to enhance the speed of SRH image acquisition and feature representation, thereby enhancing the capability of automated real-time bedside diagnosis. Specifically, we propose the VQSRS network, which integrates vector quantization with a proxy task based on patch annotation for analysis of brain tumor subtypes. Training on images collected from the SRS microscopy system, our VQSRS demonstrates a significant speed enhancement over traditional techniques (e.g., 20-30 minutes). Comparative studies in dimensionality reduction clustering confirm the diagnostic capacity of VQSRS rivals that of CNN. By learning a hierarchical structure of recognizable histological features, VQSRS classifies major tissue pathological categories in brain tumors. Additionally, an external semantic segmentation method is applied for identifying tumor-infiltrated regions in SRH images. Collectively, these findings indicate that this automated real-time prediction technique holds the potential to streamline intraoperative cancer diagnosis, providing assistance to pathologists in simplifying the process.



### 0. Main Environments

All experiments were conducted on three RTX 3080Ti GPUs.

```cmd
conda create -n vqsrs python=3.9 -y
conda activate vqsrs
pip install torch==2.0.0 torchvision==0.15.1
pip install timm==0.3.2
pip install numpy==1.23.5
pip install umap-learn==0.5.3
pip install matplotlib==3.7.1
pip install joblib==1.2.0
pip install tqdm==4.65.0
pip install seaborn==0.12.2
pip install scikit-learn==1.2.2
pip install scipy==1.9.1
```

### 1. Datasets

The data used in this study were obtained from publicly available datasets. The availability of the data can be found at https://opensrh.mlins.org/ and is openly accessible to the research community.

### 2. Train the VQSRS

- Before training, you need to modify the `homepath` and the `dataset path` in `args.py`. Ensure that the dataset format is the same as the ImageNet dataset format.

  ```cmd
  python main.py
  ```

### 3. Visualization

- Plot the UMAP visualization and perform hierarchical clustering, as well as calculate clustering metrics.

- Modify the `model_path` and `val_path` in `umap_clustermaps.py`.

  ```
  python umap_clustermaps.py
  ```

## Citation

- If you found our work useful in your research, please consider citing our works(s) at:

```latex
@article{wang2024fast,
  title={Fast Real-Time Brain Tumor Detection Based on Stimulated Raman Histology and Self-Supervised Deep Learning Model},
  author={Wang, Zijun and Han, Kaitai and Liu, Wu and Wang, Zhenghui and Shi, Chaojing and Liu, Xi and Huang, Mengyuan and Sun, Guocheng and Liu, Shitou and Guo, Qianjin},
  journal={Journal of Imaging Informatics in Medicine},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```

