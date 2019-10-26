import pathlib #To Handle Paths

import numpy as np#To Handle Arrayd
import pandas as pd #To Handle Dataframes
from sklearn.model_selection import train_test_split#To Split the Data
import joblib#To Handle Pipeline Files
import pipeline

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent #To Define the Parent Directory of the Directory Containing the Script
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models' #Path to Save the Models
DATASET_DIR = PACKAGE_ROOT / 'datasets' #Path to Import the Datasets

TESTING_DATA_FILE = DATASET_DIR / 'test.csv' #Test File Path
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv' #Train File Path
TARGET = 'SalePrice' #Model Target Variable Name

FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual',
            'OverallCond', 'YearRemodAdd', 'RoofStyle', 'MasVnrType',
            'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
            '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual',
            'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish',
            'GarageCars', 'PavedDrive', 'LotFrontage',
            # this variable is only to calculate temporal variable:
            'YrSold'] #List of Features

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = 'regression_model.pkl' #Pipeline File Name
    save_path = TRAINED_MODEL_DIR / save_file_name #Pipeline File Path
    joblib.dump(pipeline_to_persist, save_path) #Saving Pipeline File

    print('saved pipeline')


def run_training() -> None:
    """Train the model."""

    #Reading CSV Training File
    data = pd.read_csv(TRAINING_DATA_FILE)

    #Spliting Data in Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.25,
        random_state=42)  # we are setting the seed here

    #Transforming Target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    #Fitting Pipeline
    pipeline.price_pipe.fit(X_train[FEATURES],
                            y_train)

    #Saving Pipeline
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == '__main__':
    run_training()