import pandas as pd

from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset

from api import config

def capture_predictions() -> None:
    save_file='test_data_predictions.csv'
    test_data=load_dataset(file_name='test.csv')
    multiple_test_json=test_data[99:600]
    predictions=make_prediction(input_data=multiple_test_json)
    predictions_df=pd.DataFrame(predictions)
    predictions_df.to_csv(f'{config.PACKAGE_ROOT}/{save_file}')

if __name__=='__main__':
    capture_predictions()