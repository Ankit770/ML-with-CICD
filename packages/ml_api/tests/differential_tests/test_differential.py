import math
import pytest
from regression_model.config import model_config
from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset
import pandas as pd
from api import config

@pytest.mark.differential
def test_model_prediction_differential(*,save_file='test_data_predictions.csv'):
    previous_model_df=pd.read_csv(f'{config.PACKAGE_ROOT}/{save_file}')
    previous_model_predictions=previous_model_df.prediction.values
    test_data=load_dataset(file_name=model_config.TESTING_DATA_FILE)
    multiple_test_json=test_data[99:600]

    response=make_prediction(input_data=multiple_test_json)
    current_model_predictions=response.get('prediction')

    assert len(previous_model_predictions)==len(current_model_predictions)

    for previous_value,current_value in zip(previous_model_predictions, current_model_predictions):
        previous_value=previous_value.item()
        current_value=current_value.item()

        assert math.isclose(previous_value,current_value,rel_tol=model_config.ACCEPTABLE_MODEL_DIFFERENCE)