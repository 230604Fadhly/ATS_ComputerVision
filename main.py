import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


os.makedirs("results", exist_ok=True)

print("Downloading/loading dataset...")

path = kagglehub.dataset_download("crawford/emnist")
print("Dataset path:", path)

csv_path = os.path.join(path, "emnist-letters-train.csv")

data = pd.read_csv(
    csv_path,
    header=None,
    nrows=5000,
    dtype=np.uint8
)

print("Dataset loaded successfully")


balanced_data = []

# 30 sample per class (lebih ringan)
SAMPLES_PER_CLASS = 30

for label in range(1, 27):
    class_samples = data[data[0] == label]

    sampled = class_samples.sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    balanced_data.append(sampled)

balanced_data = pd.concat(balanced_data)

# Shuffle dataset
balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Balanced dataset created")
print("Total samples:", len(balanced_data))


labels = balanced_data.iloc[:, 0].values
images = balanced_data.iloc[:, 1:].values


fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    img = images[i].reshape(28, 28)

    # transpose supaya orientasi benar
    img = np.transpose(img)

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("results/sample_dataset.png")
plt.show()


print("Extracting HOG features...")

hog_features = []

for img in images:
    img = img.reshape(28, 28)
    img = np.transpose(img)
    img = img / 255.0

    feature = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    hog_features.append(feature)

X = np.array(hog_features)
y = np.array(labels)

print("HOG extraction completed")
print("Feature shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


print("Running Grid Search...")

param_grid = {
    'kernel': ['linear'],
    'C': [1, 10]
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    verbose=2,
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

print("\n===== EVALUATION RESULT =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("results/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved")
print("Classification report saved")
print("Done.")