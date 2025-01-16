

import json
import numpy as np
import pandas as pd
import mlflow.pyfunc
from xgboost import XGBClassifier
import joblib
from typing import Union, Dict, List
import logging

class Airquality_Detector(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.target_encoder = None
        self.model = None
        self.required_columns = [
            "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
            "Proximity_to_Industrial_Areas", "Population_Density"
        ]

    def load_context(self, context) -> None:
        """Load the target encoder and model from artifacts."""
        try:
            self.target_encoder = joblib.load(context.artifacts["target_encoder"])
            self.model = XGBClassifier()
            self.model.load_model(context.artifacts["model"])
            logging.info("Model and encoder loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model context: {str(e)}")
            raise

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate that all required columns are present."""
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _process_json_input(self, json_input: Union[str, Dict, List]) -> pd.DataFrame:
        """Convert JSON input to pandas DataFrame."""
        try:
            # Handle string input
            if isinstance(json_input, str):
                data = json.loads(json_input)
            else:
                data = json_input

            # Handle single record vs list of records
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("JSON input must be a dictionary or list of dictionaries")

            return df
        except Exception as e:
            raise ValueError(f"Error processing JSON input: {str(e)}")

    def _preprocess_input(self, data: Union[str, Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data regardless of format."""
        try:
            # Convert input to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = self._process_json_input(data)

            # Validate columns
            self._validate_columns(data)

            # Ensure correct column order
            data = data[self.required_columns]

            # Convert datatypes
            data = data.astype({
                "Temperature": float,
                "Humidity": float,
                "PM2.5": float,
                "PM10": float,
                "NO2": float,
                "SO2": float,
                "CO": float,
                "Proximity_to_Industrial_Areas": float,
                "Population_Density": float
            })

            return data
        except Exception as e:
            raise ValueError(f"Error preprocessing input: {str(e)}")

    def predict(self, context, model_input: Union[str, Dict, List, pd.DataFrame]) -> List[str]:
        """
        Make predictions on the input data.
        
        Args:
            context: MLflow model context
            model_input: Can be one of:
                - pandas DataFrame
                - JSON string
                - Python dictionary
                - List of dictionaries
                
        Returns:
            List of predicted air quality categories
        """
        try:
            # Preprocess the input
            processed_input = self._preprocess_input(model_input)
            # Log input for debugging
            logging.info(f"Input for prediction:\n{processed_input}") 
            # Make predictions
            predictions = self.model.predict(processed_input)
            # Convert numerical predictions to readable categories
            if self.target_encoder is not None:
                predictions = self.target_encoder.inverse_transform(predictions)
            # Log predictions for debugging
            logging.debug(f"Decoded predictions: {predictions}")
            return predictions.tolist()  # Convert numpy array to list 
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
