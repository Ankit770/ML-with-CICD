import pathlib
import regression_model

PACKAGE_ROOT=pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR=PACKAGE_ROOT / "trained_models"
DATASET_DIR=PACKAGE_ROOT / "datasets"
TESTING_DATA_FILE="test.csv"
TRAINING_DATA_FILE="train.csv"
#PIPELINE_NAME="lasso_regression"
TARGET='SalePrice'
FEATURES=['MSSubClass',
 'MSZoning',
 'Neighborhood',
 'OverallQual',
 'OverallCond',
 'YearRemodAdd',
 'RoofStyle',
 'MasVnrType',
 'BsmtQual',
 'BsmtExposure',
 'HeatingQC',
 'CentralAir',
 '1stFlrSF',
 'GrLivArea',
 'BsmtFullBath',
 'KitchenQual',
 'Fireplaces',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageCars',
 'PavedDrive',
 'LotFrontage','YrSold']
PIPELINE_SAVE_FILE="lasso_regression_output"
DROP_FEATURES='YrSold'
NUMERICAL_VARS_WITH_NA=['LotFrontage']
CATEGORICAL_VARS_WITH_NA=['MasVnrType','BsmtQual','BsmtExposure','FireplaceQu','GarageType','GarageFinish']
TEMPORAL_VARS='YearRemodAdd'
NUMERICAL_LOG_VARS=['LotFrontage','1stFlrSF','GrLivArea']
CATEGORICAL_VARS=['MSZoning',
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
 'PavedDrive']

NUMERICAL_NA_NOT_ALLOWED=[feature for feature in FEATURES if feature not in CATEGORICAL_VARS+NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED=[feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA]
ACCEPTABLE_MODEL_DIFFERENCE=0.05
