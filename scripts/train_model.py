import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train():
    df = pd.read_csv('data/processed/processed_data.csv')
    X = df.drop(['TARGET', 'ID'], axis=1, errors='ignore')
    y = df['TARGET']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    # Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_sm, y_train_sm)
    
    # Save model
    joblib.dump(clf, 'outputs/credit_risk_model.pkl')
    print("Model trained and saved to outputs/credit_risk_model.pkl")

if __name__ == "__main__":
    train()
