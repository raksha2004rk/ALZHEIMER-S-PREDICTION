# 🧠 AI Health Diagnosis (Alzheimer’s MRI Classification)

## 📌 Overview

This project focuses on detecting **Alzheimer’s Disease** using **MRI brain images** and **Deep Learning techniques**. The system classifies MRI scans into categories such as *Non-Demented* and *Moderate Demented* using a Convolutional Neural Network (CNN).

The project follows a **modular machine learning pipeline architecture**, including data preprocessing, model training, evaluation, and deployment readiness.

---

## 🚀 Project Objectives

- Develop an AI-based system for Alzheimer’s detection  
- Analyze MRI brain images  
- Build a CNN-based classification model  
- Perform preprocessing and feature extraction  
- Evaluate model performance using standard metrics  
- Create a scalable and reusable ML pipeline  

---

## 🧠 Problem Statement

Given an MRI brain scan, predict whether the patient is:

👉 **Non-Demented** or **Moderate Demented**

---

## 🗂️ Project Structure

```
AI-Health-Diagnosis/
│
├── artifacts/                 
│   ├── processed_data/
│   ├── train_data/
│   ├── test_data/
│   ├── model.h5
│   └── preprocessor.pkl
│
├── notebooks/                 
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── components/
│   │    ├── data_ingestion.py
│   │    ├── data_preprocessing.py
│   │    └── model_training.py
│   │
│   ├── pipeline/
│   │    └── training_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── app/                      
│   ├── app.py
│   └── templates/
│
├── venv/
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.10  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Matplotlib, Seaborn  
- Flask / Streamlit  

---

## 📊 Dataset Used

### ✅ OASIS MRI Dataset
- Open-access brain MRI dataset  
- Contains Normal and Demented subjects  
- Used for research in Alzheimer’s detection  

🔗 https://www.oasis-brains.org  

### ✅ Kaggle Alzheimer MRI Dataset
- Preprocessed MRI images  
- Multi-class labeled dataset  
- Used for CNN training  

🔗 https://www.kaggle.com  

---

## 🔄 ML Pipeline Workflow

### 1. Data Ingestion
- Load MRI dataset  
- Split into training and testing sets  

### 2. Data Preprocessing
- Resize images  
- Normalize pixel values  
- Data augmentation (rotation, flipping)  

### 3. Model Training
- Build CNN architecture  
- Train model using training dataset  
- Evaluate on validation dataset  

### 4. Model Evaluation
- Accuracy  
- Precision, Recall  
- Confusion Matrix  
- Grad-CAM visualization  

---

## 🧪 How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repo-link>
cd AI-Health-Diagnosis
```

### Step 2: Create & activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the pipeline

```bash
python -m src.components.data_ingestion
```

---

## 📈 Sample Output

```
artifacts/
 ├── processed_data/
 ├── train_data/
 ├── test_data/
 ├── model.h5
 └── preprocessor.pkl
```

---

## 📌 Key Highlights

- Deep Learning-based medical diagnosis  
- CNN architecture for MRI classification  
- Modular ML pipeline design  
- Explainable AI using Grad-CAM  
- Scalable and deployment-ready  

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|------|---------|
| Module not found | Activate virtual environment |
| GPU not detected | Install CUDA & cuDNN |
| Slow training | Use smaller image size |
| Model not saving | Check artifacts path |

---

## 📚 Future Improvements

- Multi-class classification  
- CT/PET scan integration  
- Web deployment  
- Cloud inference  
- Explainable AI enhancements  

---

## 👩‍💻 Author

Raksha Kadam  
B.Tech CSE (AIML)

---

## ⭐ Acknowledgements

- OASIS Dataset  
- Kaggle Dataset  
- TensorFlow Documentation  
- Research papers  

---

## 📬 Contact

Feel free to connect for queries or collaboration!
