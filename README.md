# 🧠 AI Health Diagnosis (Alzheimer’s MRI Classification)

## 📌 Overview

This project focuses on detecting **Alzheimer’s Disease** using **MRI brain images** and **Deep Learning (CNN)** techniques. The system classifies MRI scans into:

- Non-Demented  
- Moderate Demented  

It also includes **Grad-CAM visualization** for explainability.

---

## 🚀 Project Objectives

- Build an AI-based Alzheimer’s detection system  
- Perform MRI image classification using CNN  
- Implement a modular ML pipeline  
- Add explainable AI (Grad-CAM)  
- Generate training performance graphs  

---

## 🧠 Problem Statement

Given an MRI brain scan, predict whether the patient is:

👉 **Non-Demented** or **Moderate Demented**

---

## 🗂️ Project Structure

```
AI-Health-Diagnosis/
│
├── dataset/                     # Raw dataset (NOT uploaded to GitHub)
│   ├── NonDemented/
│   └── ModerateDemented/
│
├── artifacts/                  # Generated files
│   ├── train_data/
│   ├── test_data/
│   ├── model.h5
│   ├── training_plot.png
│   └── gradcam_output.jpg
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── model_training.py
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── app/
│   └── app.py                  # Streamlit app
│
├── data_split.py              # Dataset split script
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.10  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Matplotlib, Seaborn  
- Streamlit  

---

## 📊 Dataset

Download dataset from:  
👉 https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented  

### Use only:
- NonDemented  
- ModerateDemented  

### Place it inside:

```
dataset/
├── NonDemented/
├── ModerateDemented/
```

---

## 🔄 Workflow

### 1. Data Splitting

Split dataset into train/test:

```bash
python data_split.py
```

---

### 2. Model Training

```bash
python -m src.pipeline.training_pipeline
```

---

### 3. Outputs Generated

```
artifacts/
├── model.h5
├── training_plot.png
├── gradcam_output.jpg
```

---

### 4. Run Web App

```bash
streamlit run app/app.py
```

---

## 📊 Features

- CNN-based MRI classification  
- Grad-CAM explainability  
- Training accuracy & loss graphs  
- Modular pipeline structure  
- Deployment-ready UI  

---

## 📈 Sample Output

- Prediction: Non-Demented / Moderate Demented  
- Confidence score  
- Grad-CAM heatmap visualization  

---

## ⚠️ Important Notes

- Dataset is NOT uploaded to GitHub  
- Use `.gitignore` to exclude:
  ```
  dataset/
  artifacts/
  venv/
  ```

---

## 📚 Future Improvements

- Multi-class classification  
- CT/PET integration  
- Cloud deployment  
- Model optimization  
- Real-time hospital integration  

---

## 👩‍💻 Author

Raksha Kadam  
B.Tech CSE (AIML)

---

## ⭐ Acknowledgements

- Kaggle Dataset  
- OASIS Dataset  
- TensorFlow Documentation  
- Research papers on Alzheimer’s detection  

---

## 📬 Contact

Feel free to connect for queries or collaboration!