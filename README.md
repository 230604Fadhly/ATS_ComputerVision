# ATS Computer Vision  
## Handwritten Letter Classification using EMNIST Dataset with Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM)

---

# 1. Project Overview

This project is a computer vision implementation for handwritten letter recognition using the **EMNIST Letters Dataset**. The main objective of this project is to classify handwritten alphabet characters from **A to Z** by applying image feature extraction techniques and machine learning classification.

This project combines:

- **Image preprocessing**
- **Feature extraction using Histogram of Oriented Gradients (HOG)**
- **Classification using Support Vector Machine (SVM)**
- **Hyperparameter optimization using GridSearchCV**
- **Performance evaluation using classification metrics**

This project was developed as part of the **ATS Computer Vision coursework**.

---

# 2. Background

Handwritten character recognition is one of the important applications in computer vision and pattern recognition. It has practical applications in:

- Optical Character Recognition (OCR)
- Automated document processing
- Postal address recognition
- Bank check processing
- Historical manuscript digitization
- Educational handwriting analysis

Recognizing handwritten characters is challenging because of:

- Variations in writing style
- Different stroke thickness
- Character rotation
- Noise and distortion
- Similar visual appearance between characters

To address these challenges, this project uses:

### Histogram of Oriented Gradients (HOG)
HOG extracts edge and shape information from images.

### Support Vector Machine (SVM)
SVM is a supervised learning algorithm that performs well in high-dimensional feature spaces.

---

# 3. Objectives

The objectives of this project are:

## Main Objective
To build a handwritten alphabet classification model using machine learning.

## Specific Objectives

- Load and process the EMNIST Letters dataset
- Balance the dataset for fair classification
- Extract robust image features using HOG
- Train an SVM classifier
- Optimize model parameters
- Evaluate model performance
- Visualize classification results

---

# 4. Dataset Information

## Dataset Used

**EMNIST Letters Dataset**

The dataset is downloaded automatically using:

```python
kagglehub.dataset_download("crawford/emnist")
```

---

## Dataset Characteristics

| Property | Value |
|---------|------|
| Dataset Name | EMNIST Letters |
| Number of Classes | 26 |
| Labels | A-Z |
| Image Size | 28 x 28 pixels |
| Image Type | Grayscale |
| Data Format | CSV |

---

## Sampling Strategy

To reduce computational cost while maintaining class balance, the project uses:

- **30 samples per class**

Total samples:

26 classes × 30 samples = **780 samples**

Balanced sampling ensures that each class contributes equally to training.

---

# 5. Technologies and Libraries

This project uses Python and several libraries.

## Core Libraries

### Numerical Computing
- NumPy

### Data Handling
- Pandas

### Visualization
- Matplotlib
- Seaborn

### Dataset Access
- KaggleHub

### Feature Extraction
- scikit-image

### Machine Learning
- scikit-learn

---

# 6. Installation

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn kagglehub scikit-image scikit-learn
```

If multiple Python versions are installed:

```bash
py -3.11 -m pip install numpy pandas matplotlib seaborn kagglehub scikit-image scikit-learn
```

---

# 7. Project Structure

```bash
ATS_ComputerVision/
│
├── main.py
├── README.md
│
└── results/
    ├── sample_dataset.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

# 8. Methodology

## 8.1 Dataset Loading

The dataset is automatically downloaded from KaggleHub.

```python
path = kagglehub.dataset_download("crawford/emnist")
```

The CSV file is loaded into a Pandas DataFrame.

---

## 8.2 Data Balancing

The project balances the dataset by selecting:

- 30 random samples from each class

This avoids class imbalance issues.

---

## 8.3 Data Visualization

A subset of images is visualized to inspect dataset quality.

The images are reshaped from flat vectors into:

28 × 28 matrices

Because EMNIST images are rotated, transpose correction is applied.

---

## 8.4 Feature Extraction using HOG

Histogram of Oriented Gradients extracts shape and edge descriptors.

### Parameters Used

| Parameter | Value |
|----------|------|
| orientations | 9 |
| pixels_per_cell | (4,4) |
| cells_per_block | (2,2) |
| block_norm | L2-Hys |

---

### Why HOG?

HOG is chosen because it:

- Captures edge orientation
- Represents character structure effectively
- Is robust to illumination variations
- Works well for shape-based recognition

---

## 8.5 Data Splitting

Dataset split:

- **80% Training**
- **20% Testing**

Stratified splitting is used to preserve class distribution.

---

## 8.6 Classification using SVM

Support Vector Machine is selected because it performs well on high-dimensional feature vectors.

---

### Hyperparameter Optimization

Grid Search is used to find optimal parameters.

Parameter grid:

```python
param_grid = {
    'kernel': ['linear'],
    'C': [1, 10]
}
```

Cross-validation:

- **3-fold cross validation**

---

# 9. Evaluation Metrics

Model performance is measured using:

## Accuracy
Measures overall correctness.

## Precision
Measures prediction exactness.

## Recall
Measures completeness of classification.

## F1-Score
Harmonic mean of precision and recall.

## Confusion Matrix
Shows detailed class-wise prediction performance.

## Classification Report
Provides per-class metrics.

---

# 10. Workflow

The complete workflow is:

### Step 1
Download dataset

### Step 2
Load CSV data

### Step 3
Balance dataset

### Step 4
Shuffle samples

### Step 5
Visualize samples

### Step 6
Extract HOG features

### Step 7
Split training/testing data

### Step 8
Perform GridSearchCV

### Step 9
Train best SVM model

### Step 10
Evaluate model

### Step 11
Generate visual outputs

### Step 12
Save results

---

# 11. How to Run

Run the program using:

```bash
python main.py
```

or

```bash
py -3.11 main.py
```

---

# 12. Output Files

## 12.1 Sample Dataset Visualization

File:

```bash
results/sample_dataset.png
```

Purpose:

- Displays sample handwritten characters

---

## 12.2 Confusion Matrix

File:

```bash
results/confusion_matrix.png
```

Purpose:

- Visualizes classification performance

---

## 12.3 Classification Report

File:

```bash
results/classification_report.txt
```

Contains:

- Precision
- Recall
- F1-score
- Support

---

# 13. Advantages of This Approach

This implementation offers:

- Balanced training data
- Efficient feature extraction
- Fast training
- Strong classification performance
- Lightweight computational requirements

---

# 14. Limitations

Current limitations:

- Limited sample size
- Only linear kernel tested
- No deep learning comparison
- Performance depends on HOG parameter selection

---

# 15. Future Improvements

Potential enhancements:

- Increase dataset size
- Test RBF kernel
- Implement CNN
- Apply image augmentation
- Perform advanced preprocessing

---

# 16. Conclusion

This project demonstrates that combining:

- Histogram of Oriented Gradients (HOG)
- Support Vector Machine (SVM)

is an effective classical machine learning approach for handwritten character classification.

The project successfully:

- Processes handwritten image data
- Extracts meaningful features
- Classifies alphabet characters
- Evaluates performance quantitatively

This confirms that traditional computer vision methods remain effective for pattern recognition tasks.

---

# 17. Author

**Name:** 230604Fadhly  
**Course:** ATS Computer Vision  
**Semester:** Computer Vision Project  
**Institution:** Universitas

---

# 18. References

- Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
- Scikit-learn Documentation
- Scikit-image Documentation
- EMNIST Dataset Documentation
- KaggleHub Documentation