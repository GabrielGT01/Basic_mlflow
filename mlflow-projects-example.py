
import matplotlib
matplotlib.use('TkAgg')
import argparse
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import logging
import mlflow.pyfunc
import logging.config
import mlflow.sklearn
import sys
from pathlib import Path
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./updated_pollution_dataset.csv")
    parser.add_argument("--early_stopping_rounds", type=int, default=10)
    parser.add_argument("--average", choices=['micro', 'macro', 'weighted'], default='weighted')
    return parser.parse_args()

def configure_logging():
    """Configure logging handlers and return a logger instance."""
    
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )


def prepare_data(data_path):
    try:
        df = pd.read_csv(data_path)
        quality_mapping = {'Good': 0, 'Moderate': 1, 'Poor': 2, 'Hazardous': 3}
        df['Air Quality'] = df['Air Quality'].apply(lambda x: quality_mapping[x])
        # Convert Population_Density to float
        df["Population_Density"] =df["Population_Density"].astype(float)

        
        logging.debug(f"prepare_data called with data_path: {data_path}")
        return df  
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise


    
   


def split_data(loaded_data):
    logging.info("splitting data...")
    try:
        X = loaded_data.drop(columns=['Air Quality'])
        y = loaded_data['Air Quality']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        logging.debug(f"shape of X_train : {X_train.shape}")
        return X_train, X_test, X_val, y_train, y_test, y_val
    except KeyError as e:
        logging.error(f"Missing required column: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error splitting data: {e}")
        raise



def create_model(args):
    # Initialize the model for multi-class classification
    logging.info("creating model...")
    model = XGBClassifier(
        objective='multi:softmax',  # For probabilistic predictions
        num_class=4,                # Number of classes
        eval_metric=['mlogloss','merror'],   # Suitable for multi-class classification
        random_state=42,
        early_stopping_rounds=args.early_stopping_rounds
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    return model


def evaluate_model(model, y_test, y_pred, args):
    logging.info("The training finished successfully and its fitting to test dataset.")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

     # Precision and Recall
    precision = precision_score(y_test, y_pred, average=args.average)
    recall = recall_score(y_test, y_pred, average=args.average)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    logging.info("precision: %s", precision)
    logging.info("recall: %s", recall)


    return  accuracy,precision,recall


def job_done(args):
    df = prepare_data(args.data)
    if df is None:
        logging.error("Data preparation failed. Exiting the script.")
        raise RuntimeError("Data preparation failed. Please check the input file.")

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df)

    mlflow.set_experiment("a-new-demo")
    with mlflow.start_run():
        
        #Log parameters and metrics together
        params = {
            "data": args.data,
            "early_stopping_rounds": args.early_stopping_rounds,
            "average": args.average
        }
        mlflow.log_params(params)
        
        # Create and train the model
        model = create_model(args)
        trained_model = train_model(model, X_train, y_train, X_val, y_val)
        y_pred = trained_model.predict(X_test)

        accuracy, precision, recall = evaluate_model(trained_model, y_test, y_pred, args)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
        mlflow.log_metrics(metrics)

        # Corrected model input example
        model_input = pd.DataFrame([{
            "Temperature": 29.8,
            "Humidity": 59.1,
            "PM2.5": 5.2,
            "PM10": 17.9,
            "NO2": 18.9,
            "SO2": 9.2,
            "CO": 1.72,
            "Proximity_to_Industrial_Areas": 6.3,
            "Population_Density": 319.0,
        }])

        # Simulated model output
        model_output = pd.DataFrame({"Air Quality": [0]})

        # Infer signature
        signature = infer_signature(model_input, model_output)

        # Save locally using XGBoost's native format
        
        trained_model.save_model("xgboost_model.ubj")  # or Save as JSON 
        
        # Log with MLflow using XGBoost flavor
        mlflow.xgboost.log_model(
            trained_model,
            artifact_path="air-quality-xgboost-model",
            signature=signature,
            input_example=model_input
        )



if __name__ == "__main__":
    job_done(parse_args())
