import pathlib #To Handle Paths


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

def save_pipeline() -> None:
    """Persist the pipeline."""

    pass


def run_training() -> None:
    """Train the model."""

    print('Training...')


if __name__ == '__main__':
    run_training()