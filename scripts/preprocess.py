import pandas as pd
import os

def preprocess():
    # Paths
    raw_app_path = 'data/raw/application_record.csv'
    raw_credit_path = 'data/raw/credit_record.csv'
    processed_path = 'data/processed/processed_data.csv'
    
    # Load data
    app = pd.read_csv(raw_app_path)
    credit = pd.read_csv(raw_credit_path)
    
    # Data cleaning, feature engineering (Add your steps here)
    # Example: Remove duplicates
    app = app.drop_duplicates()
    credit = credit.drop_duplicates()
    
    # Encode categorical, fill missing, etc.
    # ... (your steps here)
    
    # Target variable example
    credit['default_flag'] = credit['STATUS'].apply(lambda x: 1 if x >= 2 else 0)
    default_status = credit.groupby('ID')['default_flag'].max().reset_index()
    merged = pd.merge(app, default_status, on='ID', how='left')
    merged['default_flag'].fillna(0, inplace=True)
    merged.rename(columns={'default_flag': 'TARGET'}, inplace=True)

    # Remove ID before modeling (optional)
    # merged.drop(['ID'], axis=1, inplace=True)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    merged.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    preprocess()
