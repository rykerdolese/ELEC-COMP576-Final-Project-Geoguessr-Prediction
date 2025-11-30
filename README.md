# 📍 From Pixels to Places: Street-Level Geolocation with Neural Models

**Predicting a photo’s country of origin using Google Street View imagery.**  
This project compares CNNs, fine-tuned ResNet-50, StreetCLIP zero-shot inference, and a custom StreetCLIP+MLP pipeline for large-scale geolocation modeling.

---

## 🚀 Project Overview

Inspired by Geoguessr, this project builds models that classify a street-level image into one of ~20 countries.  
We evaluate:

- CNN baseline  
- Fine-tuned ResNet-50  
- StreetCLIP zero-shot classifier  
- StreetCLIP embeddings + MLP classifier (best)  
- Multimodal caption embeddings  
- Hierarchical continent → country classifier  

The StreetCLIP+MLP model achieves **~88% test accuracy**, far outperforming all baselines.

---

## 📦 Dataset

We use the **50,000-image Geolocation (Geoguessr) dataset** from Kaggle.

- Train / Val / Test: **75% / 10% / 15%**
- Class imbalance mitigated by grouping small EU classes
- Images include rural roads, cities, signage, architecture, and landscapes

We applied:
- Targeted augmentation for minority classes  
- Manual captions for 350 images (used for multimodal experiments)

---

## 🧠 Modeling Approaches

### **1. CNN Baseline**
Simple 2-layer CNN trained from scratch.  
Result: **24% accuracy** — mostly predicts the majority class.

---

### **2. ResNet-50 (Fine-Tuned)**
ImageNet backbone, trained for 15 epochs.  
Result: **43% accuracy**.

---

### **3. StreetCLIP Zero-Shot Model**
Zero-shot classification is performed by comparing each image embedding to text prompts such as:

- "a photo taken in France"  
- "a photo taken in Japan"  
- "a photo taken in Brazil"  

**Performance:**  
- **73% Top-1 accuracy**  
- **91% Top-3 accuracy**

---

### **4. StreetCLIP + MLP (Best Model)**

Pipeline:
1. Extract 768-dim StreetCLIP embeddings  
2. Train a 2-layer MLP (256 → 128 → softmax)  
3. Train 200 epochs with Adam  

**Results:**

| Model | Val Acc | Test Acc |
|-------|---------|----------|
| CNN Baseline | 24% | 24% |
| ResNet-50 | 43% | 43% |
| StreetCLIP Zero-Shot | — | 73% |
| **StreetCLIP + MLP** | **88.8%** | **88.0%** |

---

### **5. Multimodal Caption Embeddings**
350 images manually captioned and embedded with SentenceTransformer (MiniLM).  
Findings:

- On full dataset → captions **do not boost results**
- On tiny datasets → captions improve accuracy **dramatically** (8% → 44%)

---

### **6. Hierarchical Model (Continent → Country)**  
This approach underperforms because continent misclassification cascades.  
Test accuracy: **~60.5%**.

---

## 📊 Results Summary

StreetCLIP-based models dominate performance. CNN and ResNet models struggle with fine-grained geographic cues, while StreetCLIP embeddings contain rich spatial information even without training.

Grad-CAM visualizations reveal that the model attends to:
- vegetation  
- signage  
- road markings  
- architectural structure  

---

## 📁 Repository Structure


````text
.
├── modeling_cleaned.ipynb
├── prepare_geo_dataset.py
├── streetclip_embedding_generation.py
├── resnet50_trained.pt
├── train_dataset.csv
├── val_dataset.csv
├── test_dataset.csv
├── image_captions.txt
├── plots/
└── README.md
