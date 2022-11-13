import os.path

import pandas as pd

from zeuspy.src.model import Model


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise ValueError(f'The provided file path is invalid: {file_path}')
    try:
        load_data.data = pd.read_csv(file_path)
        print('Data is loaded successfully!')
    except Exception as e:
        raise ValueError(f'{e.value}\nFailed to load data from: {file_path}')


def train_model(data: pd.DataFrame, model_type: str, algorithm: str, target_var: str) -> None:
    print(f"Training a {model_type} model using {algorithm} with target variable {target_var}..."
          "This might take a while ⏳")
    model = Model(model_type, target_var, algorithm=algorithm)
    model.train(data)
    print(f"Model training finished ✅")
