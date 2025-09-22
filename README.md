
## Semantic Segmentation of 3D Point Clouds

### 📌 Project Overview
The **Semantic_Segmentation** project provides a complete pipeline for  indoor 3D point cloud data** (laboratory/university environment).  
It supports **PointNet++** and **RandLA-Net** models for per-point class label prediction across **14 classes**.  

The pipeline handles:
- Data preprocessing (conversion, normalization, rotation, label mapping, chunking)
- Model training and evaluation
- Automatic experiment reproducibility using **DVC**
- End-to-end automation from raw `.ply` files to trained models and predictions

---


## ⚙️ Dependencies

### 1) Dependencies list which are listed in requirements.txt. Install them with:

```bash
matplotlib

seaborn

pandas

scikit-learn

open3d

plyfile

python-box

joblib

pyyaml

dvc
```
### 2) externally install pytorch version with gpu



---
# 🛠️ Installation & Setup

## 1. Clone the repository
```bash
git clone https://github.com/DikshantPatel2210/Semantic_Segmentation.git
cd Semantic_Segmentation
```


## 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows
```
##3. Install dependencies


```bash
pip install -r requirements.txt
pip install torch==2.7.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. Prepare dataset structure

### Before running the pipeline, ensure your dataset is organized as follows:
```bash
artifacts/
├── train/   # contains .ply training files
└── test/    # contains .ply testing files
```

## 5. Initialize DVC
```bash
dvc init
```

## ✅ Running the Pipeline

Before running the pipeline, make sure to **update the `params.yaml` and `config/config.yaml` files** according to your dataset and training preferences.

### 1. Validate Setup
Check that the dataset structure and necessary folders are correct:

```bash
dvc repro directory_structure_validation

dvc repro check_gpu_connection
```
## 2. Preprocessing steps
```bash
dvc repro ply_to_npy
dvc repro rotation  (if needed)
dvc repro normalization
dvc repro label_mapping
dvc repro chunks_process
```
### 4 Training & Testing

### Pointnet++
```bash
dvc repro pointnet2_input  
dvc repro pointnet2_model
dvc repro pointnet2_train
dvc repro pointnet2_test
```

### RandlaNet
```bash
dvc repro randlanet_input
dvc repro randlanet_model
dvc repro randlanet_train
dvc repro randlanet_test
```

## 📊 Outputs

### After training/testing, the pipeline generates:

 ✅  Trained model weights (.pth)

 ✅  Confusion matrix visualizations

 ✅  Predictions in CSV format with per-point predicted labels

 ✅  Organized results accordingly  in:

```bash
output_pointnet2/
output_randlanet/
```


## 📖 References

1. **PointNet++:** Qi, Charles R., et al. *Deep Hierarchical Feature Learning on Point Sets in a Metric Space.*  
   [Paper Link](https://arxiv.org/abs/1706.02413)

2. **RandLA-Net:** Hu, Qiang, et al. *RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds.*  
   [Paper Link](https://arxiv.org/abs/1911.11236)




## License
This project is licensed under the Apache License, Version 2.0, January 2004. See [http://www.apache.org/licenses/](http://www.apache.org/licenses/) for more details.
