# Credit Risk Prediction Project

## Project Overview

This project focuses on developing a robust predictive model to identify potential loan defaulters among credit applicants. The goal is to help financial institutions effectively manage credit risk and make informed, data-driven loan approval decisions to minimize losses and optimize portfolio health.

---

## Key Components

- **Data Cleaning and Feature Engineering:**  
  Handling missing values, encoding categorical and boolean variables, and preparing the dataset for modeling.

- **Exploratory Data Analysis (EDA):**  
  Understanding data distributions, detecting class imbalance, and identifying critical features impacting credit risk.

- **Handling Class Imbalance:**  
  Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training dataset and improve minority class prediction.

- **Model Development:**  
  Training Random Forest classifiers on balanced and sampled datasets, optimizing for recall and precision to better identify defaulters.

- **Model Evaluation:**  
  Comprehensive evaluation with classification reports, confusion matrix heatmaps, ROC-AUC scores, and detailed visualizations for model interpretation.

- **Prediction and Business Decision Logic:**  
  Generating approval/rejection decisions based on predicted default probabilities, enabling practical credit risk mitigation.

- **Results Visualization and Reporting:**  
  Key plots and summary tables presenting findings clearly to both technical and non-technical stakeholders.

- **Code Organization and Reproducibility:**  
  Modular code structure with separate folders for raw and processed data, notebooks, scripts, outputs, and a requirements file for easy environment setup.

---

## Business Context

Loan default poses a significant financial risk to lending institutions. By leveraging advanced machine learning techniques alongside rigorous data preprocessing and class balancing, this project automates and enhances the loan approval process. This reduces human bias, streamlines decision-making, and safeguards financial health.

---

## Repository Structure

credit_risk_project/
│
├── data/
│ ├── raw/ # Original, unmodified data files
│ └── processed/ # Cleaned and prepared datasets for modeling
│
├── notebooks/
│ └── credit_risk_analysis.ipynb # Notebook with analysis and modeling steps
│
├── scripts/
│ ├── preprocess.py # Data cleaning and preparation script
│ ├── train_model.py # Model training and saving script
│ ├── predict.py # Prediction using trained model
│ └── evaluate.py # Optional: Evaluation and visualization script
│
├── outputs/
│ ├── figures/ # Saved plots and visualization images
│ └── predictions/ # CSV files containing model predictions and results
│
├── README.md # Project overview and instructions (this file)
└── requirements.txt # Python package dependencies

---

## How to Run

1. Clone the repository
git clone (https://github.com/Gourav-2003/credit_risk_project)
cd credit_risk_project

2. Install required packages:  
pip install -r requirements.txt

3. Run data preprocessing (optional step if data not preprocessed):  
python scripts/preprocess.py

4. Train the model:  
python scripts/train_model.py

5. Make predictions on new data or test set:  
python scripts/predict.py

6. Evaluate the model (optional):  
python scripts/evaluate.py

7. Alternatively, run the Jupyter notebook for full analysis and modeling:  
jupyter notebook notebooks/credit_risk_analysis.ipynb


---

## Contact

For any questions or suggestions, please contact: gouravmuchhal476@gmail.com

