from typing import Tuple

import pandas as pd


VALID_MODEL_TYPES = ('regression', 'classification')
VALID_ALGORITHMS = {'regression': ('Linear Regression',),
                    'classification': ('Logistic Regression',
                                       'SVM',
                                       'Random Forest')}


class Model:
    """The core class for training and testing models"""

    def __init__(self, model_type, algorithm=None, model_path=None):
        self.model_type = model_type.lower()
        # `algorithm` won't matter if user wants to use a saved model,
        # only the model_type is required.
        self.algorithm = algorithm.lower() if algorithm else self._default_algo()
        if model_path:
            # load the model from disk
            self.model = self._load_model_object(model_path)
        else:
            # going to build a new model
            self.model = None
        self._validate_model_options()

    def _load_model_object(self, model_path):
        """Load the scikit-learn model from the given path"""
        raise NotImplementedError

    def _default_algo(self) -> str:
        if self.model_type == 'regression':
            return 'linear_regression'
        elif self.model_type == 'classification':
            return 'logistic regression'
        else:
            raise NotImplementedError(f'{self.model_type} is currently not supported.')

    def _validate_model_options(self):
        """Ensure the model type and algorithm are supported"""
        assert self.model_type in VALID_MODEL_TYPES
        assert self.algorithm in VALID_ALGORITHMS[self.model_type]

    def _train_test_split(self, data, ratio=0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets based on the ratio"""
        raise NotImplementedError

    def _train_linear_regression(self, df_train):
        raise NotImplementedError

    def _train_logistic_regression(self, df_train):
        raise NotImplementedError

    def _regression_model(self, algo_name, df_train):
        raise NotImplementedError

    def _classifier(self, algo_name, df_train):
        raise NotImplementedError

    def evaluate(self, df_test):
        """Evaluate the model performance on the given data set"""
        # should only require self.model and self.model_type
        raise NotImplementedError

    def save_model(self, save_to=None):
        """Save model object to disk"""
        raise NotImplementedError

    def train(self, data, save_to=None):
        df_train, df_test = self._train_test_split(data)
        if self.model_type == 'regression':
            self.model = self._regression_model(self.algorithm, df_train)
        else:
            self.model = self._classifier(self.algorithm, df_train)

        self.evaluate(df_test)

        # save the trained model
        self.save_model(save_to)
