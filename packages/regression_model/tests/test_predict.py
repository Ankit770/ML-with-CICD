from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset
import json

import math

def test_make_single_prediction():
    test_data=load_dataset(file_name='test.csv')
    single_test_json=test_data[0:1].to_json(orient='records')

    subject=make_prediction(input_data=json.loads(single_test_json))

    assert subject is not None
    assert isinstance(subject.get('prediction')[0],float)
    assert math.ceil(subject.get('prediction')[0])==112476

def test_make_multiple_prediction():
    test_data=load_dataset(file_name='test.csv')
    original_data_length=len(test_data)
    multiple_test_json=test_data.to_json(orient='records')
    subject=make_prediction(input_data=json.loads(multiple_test_json))
    assert subject is not None
    assert len(subject.get('prediction'))==1451
    assert len(subject.get('prediction'))!=original_data_length


