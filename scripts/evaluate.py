import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate():
    model = joblib.load('outputs/credit_risk_model.pkl')
    df = pd.read_csv('data/processed/processed_data.csv')
    X = df.drop(['TARGET', 'ID'], axis=1, errors='ignore')
    y = df['TARGET']
    
    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate()
