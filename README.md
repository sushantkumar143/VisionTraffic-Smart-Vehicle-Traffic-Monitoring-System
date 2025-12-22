# Advanced Vehicle Analytics & Traffic Violation Detection System

[cite_start]This repository contains the source code for an Intelligent Transportation System (ITS) developed as a Predictive Analytics project at Lovely Professional University[cite: 1, 16, 17]. [cite_start]The system leverages computer vision to monitor traffic, estimate vehicle speed, and detect violations in real-time[cite: 4, 69, 72].

---

## 🚀 Key Features

* [cite_start]**Multi-Class Object Detection**: Detects and classifies 7 distinct classes: Car, Truck, Bus, Auto, Two-Wheeler, Plate, and Blur-Plate[cite: 79].
* [cite_start]**Real-Time Speed Estimation**: Calculates velocity based on the Euclidean distance of centroids between frames[cite: 80, 570].
* [cite_start]**Anomaly Detection**: Automatically flags overspeeding vehicles based on a configurable threshold (e.g., >80 km/h)[cite: 80, 571].
* [cite_start]**Automatic License Plate Recognition (ALPR)**: Integrates OCR to read number plates of violating vehicles[cite: 81, 637].
* [cite_start]**Strategic Reporting**: Generates automated PDF reports summarizing traffic flow dynamics and model robustness[cite: 85, 634, 635].

---

## 🛠️ Technology Stack

* [cite_start]**Model**: YOLOv8 (Nano, Small, and Medium variants)[cite: 82, 197].
* [cite_start]**Framework**: Ultralytics YOLOv8[cite: 200, 644].
* [cite_start]**Interface**: Streamlit Dashboard[cite: 85, 568].
* [cite_start]**Database**: MySQL & MS Excel for violation logging[cite: 85, 633].
* [cite_start]**OCR**: EasyOCR[cite: 647].

---

## 📊 Model Performance & Comparison

[cite_start]We evaluated three variants of the YOLOv8 model to find the optimal balance between accuracy and inference speed[cite: 82, 433].

| Model Variant | Parameters (M) | mAP@50 (Accuracy) | Inference Speed (ms) | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n (Nano)** | 3.2 M | 0.829 | 6.5 ms | [cite_start]**Selected**: Fastest, real-time capable [cite: 435, 439] |
| **YOLOv8s (Small)** | 11.2 M | 0.841 | 12.8 ms | [cite_start]Good accuracy, slower [cite: 435] |
| **YOLOv8m (Medium)** | 25.9 M | 0.855 | 22.4 ms | [cite_start]Best accuracy, too slow for CPU [cite: 435] |



### Best Model Metrics (YOLOv8n)
* [cite_start]**Precision**: ~0.915 (High confidence in True Positives)[cite: 443].
* [cite_start]**Recall**: ~0.90 (Low False Negatives)[cite: 444].
* [cite_start]**mAP@50**: 0.829[cite: 445].



---

## 🖼️ Dataset & Preprocessing

[cite_start]The model was trained on a dataset of ~960 images containing diverse traffic scenarios[cite: 108].
* [cite_start]**Source**: Roboflow Universe & Kaggle[cite: 90, 648].
* [cite_start]**Augmentations**: Applied Mosaic Augmentation, HSV Scaling, and Random Flips to ensure model robustness under different lighting conditions[cite: 114, 117, 118, 119].



---

## 💻 Dashboard Preview

[cite_start]The Streamlit interface provides a "Speed Limit Setter" and a live "Annotated Output" view[cite: 582, 597, 621].



---

## 🔮 Future Scope

* [cite_start]**Night Vision**: Training on thermal or low-light datasets[cite: 639].
* [cite_start]**Edge Deployment**: Porting to Raspberry Pi or NVIDIA Jetson Nano[cite: 640].
* [cite_start]**Predictive Accident Modeling**: Using LSTM networks to predict collisions based on trajectories[cite: 641].
* [cite_start]**Security Features**: Adding helmet detection and triple riding alerts[cite: 642].

---

## 👥 Contributors
* [cite_start]**Student**: Sushant Kumar (Reg No. 12311087) [cite: 7, 8]
* [cite_start]**Mentor**: Dr. Tanima Thakur [cite: 12]

**Project Links**: [GitHub](https://github.com/sushantkumar143/VisionTraffic-Smart-Vehicle-Traffic-Monitoring-System) | [cite_start][LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7406932700533551105/) [cite: 650, 651]
