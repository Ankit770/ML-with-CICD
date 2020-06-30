import numpy as np
from sklearn.model_selection import train_test_split
from regression_model import pipeline
from regression_model.processing.data_management import load_dataset,save_pipeline
from regression_model.config import config
from regression_model import __version__ as _version
import logging

# def save_pipeline(*, pipeline_to_persist) -> None:
#     save_file_name="regression_model.pkl"
#     save_path=config.TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(pipeline_to_persist, save_path)
#     print("saved_pipeline")
_logger=logging.getLogger(__name__)
def run_training() -> None:
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    X_train,X_test,y_train,y_test=train_test_split(data[config.FEATURES],data[config.TARGET],test_size=0.1,random_state=0)
    y_train=np.log(y_train)
    pipeline.price_pipe.fit(X_train[config.FEATURES],y_train)
    _logger.info(f"saving model version: {_version}")   
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)
    print("Training Completed")

if __name__=='__main__':
    run_training()