# ATS Computer Vision - EMNIST Letter Classification using HOG + SVM

## Deskripsi Project
Project ini merupakan implementasi klasifikasi huruf tulisan tangan menggunakan dataset **EMNIST Letters** dengan pendekatan:

- **Feature Extraction:** Histogram of Oriented Gradients (HOG)
- **Classification Model:** Support Vector Machine (SVM)
- **Hyperparameter Tuning:** Grid Search Cross Validation

Tujuan project ini adalah mengklasifikasikan karakter huruf A-Z dari dataset EMNIST menggunakan ekstraksi fitur citra dan machine learning.

---

## Dataset
Dataset yang digunakan adalah:

**EMNIST Letters Dataset**

Dataset diunduh secara otomatis menggunakan:

```python
kagglehub.dataset_download("crawford/emnist")