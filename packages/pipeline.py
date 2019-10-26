from sklearn.pipeline import Pipeline #To Create a Pipeline with a List of Steps of Transformations

import preprocessors as pp


CATEGORICAL_VARS = ['MSZoning',
                    'Neighborhood',
                    'RoofStyle',
                    'MasVnrType',
                    'BsmtQual',
                    'BsmtExposure',
                    'HeatingQC',
                    'CentralAir',
                    'KitchenQual',
                    'FireplaceQu',
                    'GarageType',
                    'GarageFinish',
                    'PavedDrive'] #List of Categorical Features

PIPELINE_NAME = 'lasso_regression' #Name of The Pipeline

price_pipe = Pipeline(
    steps=[
            ('categorical_imputer', pp.CategoricalImputer(variables=CATEGORICAL_VARS)), #Removing NaNs from Categorical Variables
        ])