# Pattern-Recognition-Computer-Vision

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import yaml

data_yaml = {
    "path": "/content/dataset_final",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",

    "names": [
        'Chili___Anthracnose_fruit',
        'Chili___Bacterial_leaf_spot',
        'Chili___Healthy_fruit',
        'Chili___Healthy_leaf',
        'Chili___Mosaic_virus_leaf',
        'Eggplant___Cercospora_leaf_spot',
        'Eggplant___Colorado_potato_beetle',
        'Eggplant___Fruit_rot',
        'Eggplant___Healthy_fruit',
        'Eggplant___Healthy_leaf',
        'Potato___Alternaria_solani_leaf',
        'Potato___Common_scab_fruit',
        'Potato___Healthy_fruit',
        'Potato___Healthy_leaf',
        'Potato___Phytopthora_infestans_leaf',
        'Tomato___Antrhacnose_fruit',
        'Tomato___Bacterial_spot_leaf',
        'Tomato___Early_blight_leaf',
        'Tomato___Healthy_fruit',
        'Tomato___Healthy_leaf',
        'Tomato___Late_blight_leaf',
        'Tomato___Leaf_mold',
        'Tomato___Tomato_yellow_leaf_curl_virus'
    ]
}

with open("/content/dataset_final/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("data.yaml")
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/content/dataset_final/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,

    device=0,

    optimizer="AdamW",
    lr0=0.001,

    patience=10,

)
from ultralytics import YOLO

# Augmentation Settings
aug_params = {
    "fliplr": 0.0,
    "flipud": 0.0,

    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,

    "degrees": 10,
    "scale": 0.5,
    "translate": 0.1,

    "mosaic": 1.0
}

model = YOLO("yolov8s.pt")

# experiment / training
results = model.train(
    data="/content/dataset_final/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,

    **aug_params,

    project="augmentation_experiments",
    name="exp_no_flip",
    save=True
)

print("Experiment finished")
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load model
model = YOLO("/content/best (4).pt")

# Predict on external test images
model.predict(
    source="/content/test_img",
    save=True,
    conf=0.25
)

# Show prediction results
predict_path = "/content/runs/detect/predict"

for img_name in os.listdir(predict_path):

    if img_name.endswith(('.jpg', '.png', '.jpeg')):

        img_path = os.path.join(predict_path, img_name)

        image = Image.open(img_path)

        plt.figure(figsize=(10,8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(img_name)
        plt.show()

# Evaluation Metrics
results = model.val(data="/content/dataset_final/data.yaml")

print("\nEvaluation Metrics:\n")

print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load model
model = YOLO("/content/best (5).pt")

# Predict on external test images
model.predict(
    source="/content/test_img",
    save=True,
    conf=0.25
)

# Show prediction results
predict_path = "/content/runs/detect/predict"

for img_name in os.listdir(predict_path):

    if img_name.endswith(('.jpg', '.png', '.jpeg')):

        img_path = os.path.join(predict_path, img_name)

        image = Image.open(img_path)

        plt.figure(figsize=(10,8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(img_name)
        plt.show()

# Evaluation Metrics
results = model.val(data="/content/dataset_final/data.yaml")

print("\nEvaluation Metrics:\n")

print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")
import os
import cv2
import pandas as pd
from ultralytics import YOLO

model = YOLO("/content/best (5).pt")

folder_path = "/content/dataset_final/test/images"

output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

data = []

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w, _ = image.shape
    image_area = h * w

    results = model(image)
    boxes = results[0].boxes

    count = len(boxes)
    total_disease_area = 0

    if count == 0:
        level = "No Disease"
    else:
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            label_name = model.names[cls]

            if "Healthy" in label_name:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            total_disease_area += box_area

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(box.conf[0])

            label = f"{label_name} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if total_disease_area == 0:
            level = "No Disease"
        else:
            severity = total_disease_area / image_area

            if severity < 0.1:
                level = "Mild"
            elif severity < 0.3:
                level = "Moderate"
            else:
                level = "Severe"

    cv2.putText(image, f"Count: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Severity: {level}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, image)

    data.append([img_name, count, level])

df = pd.DataFrame(data, columns=["Image", "Count", "Severity"])

print("\nResults Table:\n")
print(df)

df.to_csv("results.csv", index=False)
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

results_folder = "results"

no_disease = df[df["Severity"] == "No Disease"].head(2)
mild = df[df["Severity"] == "Mild"].tail(1)
moderate = df[df["Severity"] == "Moderate"].tail(1)
severe = df[df["Severity"] == "Severe"].tail(1)
selected = pd.concat([no_disease, mild, moderate , severe])

plt.figure(figsize=(10, 6))

for i, (_, row) in enumerate(selected.iterrows()):
    img_name = row["Image"]
    severity = row["Severity"]

    img_path = os.path.join(results_folder, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print("Error loading:", img_name)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(severity)
    plt.axis("off")

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

severity_counts = df["Severity"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(severity_counts.index, severity_counts.values)

plt.title("Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Images")

plt.show()
plt.figure(figsize=(6,6))

plt.pie(
    severity_counts.values,
    labels=severity_counts.index,
    autopct='%1.1f%%'
)

plt.title("Severity Percentage Distribution")
plt.show()
import seaborn as sns

plt.figure(figsize=(6,4))
sns.boxplot(x="Severity", y="Count", data=df)

plt.title("Object Count per Severity Level")
plt.show()
