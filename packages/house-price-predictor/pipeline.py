from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline #To Create a Pipeline with a List of Steps of Transformations
from sklearn.preprocessing import MinMaxScaler #To Scale the Features

import preprocessors as pp

# List of Categorical Features with NA in Train Set
CATEGORICAL_VARS_WITH_NA = [
    'MasVnrType', 'BsmtQual', 'BsmtExposure',
    'FireplaceQu', 'GarageType', 'GarageFinish'
]

# List of Numerical Features with NA in Train Set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

TEMPORAL_VARS = 'YearRemodAdd'

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = 'YrSold'

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# List of Categorical Features
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']

price_pipe = Pipeline(
    steps=[
        #Step 1 - Removing NA Values from Categorical Features
        ('categorical_imputer', pp.CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
        #Step 2 - Removing NA Values from Numerical Features
        ('numerical_inputer', pp.NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        #Step 3 -
        ('temporal_variable', pp.TemporalVariableEstimator( variables=TEMPORAL_VARS, reference_variable=TEMPORAL_VARS)),
        #Step 4 -
        ('rare_label_encoder', pp.RareLabelCategoricalEncoder( tol=0.01, variables=CATEGORICAL_VARS)),
        #Step 5 -
        ('categorical_encoder', pp.CategoricalEncoder(variables=CATEGORICAL_VARS)),
        #Step 6 -
        ('log_transformer', pp.LogTransformer(variables=NUMERICALS_LOG_VARS)),
        #Step 7 -
        ('drop_features', pp.DropUnecessaryFeatures(variables_to_drop=DROP_FEATURES)),
        #Step 8 - Scaling Features
        ('scaler', MinMaxScaler()),
         #Step 9 - Fitting Model
        ('Linear_model', Lasso(alpha=0.005, random_state=42))
        ])