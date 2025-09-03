import pandas as pd
import joblib

def predict():
    model = joblib.load('outputs/credit_risk_model.pkl')
    df = pd.read_csv('data/processed/processed_data.csv')
    X = df.drop(['TARGET', 'ID'], axis=1, errors='ignore')
    
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df['Predicted'] = preds
    df['Predicted_Prob'] = probs
    
    df.to_csv('outputs/predictions/predicted_results.csv', index=False)
    print("Predictions saved to outputs/predictions/predicted_results.csv")

if __name__ == "__main__":
    predict()
