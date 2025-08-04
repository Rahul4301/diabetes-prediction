
# 🩺 Diabetes Prediction Suite

Advanced diabetes risk prediction using deep learning (PyTorch), XGBoost, and modern data science tools. Includes interactive Streamlit UI, model training, evaluation, and visualizations.

---

## Features

- **Streamlit App**: Modern UI for risk prediction and data exploration (`st.py`)
- **Deep Learning & XGBoost Models**: Training, evaluation, and inference scripts (`d_t.py`, `test.py`)
- **Data Preprocessing**: Imputation, scaling, and feature engineering
- **Visualizations**: Feature importance, correlation matrix, training loss
- **Large Dataset**: Extended from original Kaggle Pima dataset to 50,000+ samples

---

## Quick Start

1. **Install requirements**
   ```bash
   pip install -r hackathon123/requirements.txt
   ```
2. **Run Streamlit app**
   ```bash
   streamlit run hackathon123/st.py
   ```
3. **Train or test models**
   - Deep learning: `python hackathon123/d_t.py`
   - XGBoost test: `python hackathon123/test.py`

---

## Folder Structure

```
hackathon123/
├── d_t.py                  # Deep learning model training (PyTorch)
├── st.py                   # Streamlit web app
├── test.py                 # XGBoost model evaluation
├── requirements.txt        # Python dependencies
├── diabetes.csv            # Sample dataset (Pima Indians)
├── diabetes1.csv           # Extended dataset (50,000+ rows)
├── xgboost_diabetes_model.pkl  # Pretrained XGBoost model
├── model/
│   ├── best_model.pt           # Best PyTorch model
│   ├── diabetes_model.pt       # PyTorch model
│   ├── model_architecture.pt   # Model architecture
│   ├── preprocessor.pkl        # Preprocessing pipeline
│   ├── feature_names.pkl       # Feature names
│   ├── correlation_matrix.png  # Correlation heatmap
│   ├── feature_importance.png  # Feature importance plot
│   └── training_loss.png       # Training loss curve
└── ...
```

---

## Requirements

- Python 3.8+
- See `hackathon123/requirements.txt` for full list

---

## Credits

- Data: [Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Authors: Rahul Suthar, Hackathon Team
