# Deep Learning and Generative AI


# 🚗 Revolutionizing Car Damage Detection  
### Deep Learning and Generative AI for 3D Analysis & Automated Claims Processing

![ResNet50 architecture](https://upload.wikimedia.org/wikipedia/commons/2/2d/Residual_block.png)

## 📌 Project Overview

This project presents a deep learning pipeline using **ResNet50** for multi-class vehicle damage classification across six categories (front/rear breakage, crushed, and normal). It is built to automate insurance claim evaluations and vehicle inspections by drastically reducing assessment time and human error.

- 📈 Achieved **85% overall accuracy**
- 🔁 Implemented **data augmentation** for generalization
- ⚙️ Deployed with **PyTorch**, ready for ONNX or cloud deployment
- 🤖 Follows **CRISP-DM methodology**

---

## 📂 Dataset

- **Total Images**: 2,300  
- **Classes**:
  - `F_Breakage`, `F_Crushed`, `F_Normal`
  - `R_Breakage`, `R_Crushed`, `R_Normal`

Sample image preprocessing:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 🧠 Model Architecture

- Base Model: **ResNet50 (pre-trained on ImageNet)**
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam` (lr=0.005)
- Epochs: 50–100 (tuned per class performance)

---

## 📊 Evaluation

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| F_Normal     | 0.94      | 0.94   | 0.94     |
| R_Crushed    | 0.80      | 0.76   | 0.78     |
| R_Normal     | 0.88      | 0.92   | 0.90     |
| **Macro Avg**| —         | —      | **0.84** |
| **Accuracy** | —         | —      | **85%**  |

---

## 🚀 Deployment

- Model saved as: `car_damage_detection_model.pth`
- Inference-ready: Real-time predictions for insurance APIs
- Future scope:
  - ONNX conversion for edge devices
  - GPT-based report generation
  - 3D damage estimation via VoxelNet

---

## 🔍 Research Objectives

- 🧠 Use deep learning (CNN + transfer learning) for car damage classification
- 🔁 Improve robustness with data augmentation
- 📉 Optimize classification accuracy using standard metrics (Precision, Recall, F1)

---

## 📌 Future Improvements

- Expand dataset to **10,000+ images**
- Integrate **3D imaging** for depth-aware analysis
- Use **Transformers** for attention-focused classification
- Develop **edge/cloud deployment** modules

---

## 🧑‍💻 Authors

- **Shivansh Bhatnagar** 
- **Malav Naik**   
- **Pratik Sunar**    
*National College of Ireland – MSc in Data Analytics*

---

## 📜 License

This project is for academic purposes. Please contact authors for commercial use.