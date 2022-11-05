from dataclasses import dataclass
from typing import Callable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


VALID_MODEL_TYPES = ('regression', 'classification')
VALID_ALGORITHMS = {'regression': ('Linear Regression',),
                    'classification': ('Logistic Regression',
                                       'SVM',
                                       'Random Forest')}


class Model:
    """The core class for training and testing models"""

    def __init__(self, model_type, target_var, algorithm=None, model_path=None):
        self.model_type = model_type.lower()
        # `algorithm` won't matter if user wants to use a saved model,
        # only the model_type and target_var are required in that case
        self.algorithm = algorithm
        self.target_var = target_var
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

    def _validate_model_options(self):
        """Ensure the model type and algorithm are supported"""
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Model type must be one of the following: {VALID_MODEL_TYPES}")
        if not self.model and not self.algorithm:
            raise ValueError("Algorithm is required if not using an existing model")
        if self.algorithm not in VALID_ALGORITHMS[self.model_type]:
            raise NotImplementedError(f'{self.algorithm} is currently not supported.')

    def _train_test_split(self, data, ratio=0.7, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets based on the ratio"""
        X = data.loc[:, data.columns != self.target_var]
        y = data[self.target_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=seed)
        print(f"Splitting data into training set (shape: {X_train.shape}) and testing set (shape: {X_test.shape})")

        return X_train, X_test, y_train, y_test

    def _linear_regression(self, X, y):
        try:
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.evaluate_model(X, y, 'training')
        except Exception as e:
            raise RuntimeError(f"Unknown error occurred when training the linear regression model: {e}")

    def _logistic_regression(self, X, y):
        try:
            self.model = LogisticRegression()
            self.model.fit(X, y)
            self.evaluate_model(X, y, 'training')
        except Exception as e:
            raise RuntimeError(f"Unknown error occurred when training the logistic regression model: {e}")

    def _train_regression_model(self, algo_name, X, y):
        if algo_name == 'Linear Regression':
            self._linear_regression(X, y)
        else:
            raise NotImplementedError

    def _train_classifier(self, algo_name, X, y):
        if algo_name == 'Logistic Regression':
            self._logistic_regression(X, y)
        else:
            raise NotImplementedError

    def evaluate_model(self, X, y, apply_on):
        """Evaluate the model performance on the given data set"""
        y_pred = self.model.predict(X)
        if self.model_type == 'regression':
            evaluator = RegressionModelEval(apply_on=apply_on)
        else:
            evaluator = ClassifierEval(apply_on=apply_on)
        evaluator.calc_metrics(y, y_pred)
        evaluator.display()

    def save_model(self, save_to=None):
        """Save model object to disk"""
        if save_to:
            # TODO: Validate the provided path
            joblib.dump(self.model, save_to)
            print(f"Saved model object to {save_to}")

    def train(self, data, save_to=None):
        """Train the model on the training set and evaluate using the testing set"""
        X_train, X_test, y_train, y_test = self._train_test_split(data)
        if self.model_type == 'regression':
            self._train_regression_model(self.algorithm, X_train, y_train)
            self.evaluate_model(X_test, y_test, 'testing')
        else:
            self._train_classifier(self.algorithm, X_train, y_train)
            self.evaluate_model(X_test, y_test, 'testing')

        if not save_to:
            suffix = '20221101'  # TODO: use today's date yyyymmdd format
            save_to = f'{self.model_type}_{self.target_var}_{suffix}.pkl'
        self.save_model(save_to)


@dataclass
class Metric:
    name: str
    value: float = np.inf
    formula: Callable = None

    def calculate(self, y, y_pred):
        self.value = self.formula(y, y_pred)


class Evaluator:

    def __init__(self, apply_on):
        self.apply_on = apply_on
        self.metric_list = []

    def calc_metrics(self, y, y_pred):
        for metric in self.metric_list:
            metric.calculate(y, y_pred)

    def display(self):
        print("#" * 10 + f" Model Results ({self.apply_on} set) " + "#" * 10)
        for metric in self.metric_list:
            print(f'{metric.name}: {metric.value:.4f}')


class RegressionModelEval(Evaluator):

    def __init__(self, apply_on):
        super().__init__(apply_on)
        self.model_type = 'regression'
        mse = Metric('Mean Squared Error', formula=mean_squared_error)
        r2 = Metric('R squared', formula=r2_score)
        self.metric_list = [mse, r2]

    def __repr__(self):
        return f"RegressionModelEval <model_type={self.model_type}, apply_on={self.apply_on}," \
               f"metrics={', '.join(m.name for m in self.metric_list)}>"


class ClassifierEval(Evaluator):

    def __init__(self, apply_on):
        super().__init__(apply_on)
        self.model_type = 'classification'
        acc = Metric('Accuracy', formula=accuracy_score)
        f1 = Metric('F1 score', formula=f1_score)
        roc_auc = Metric('ROC AUC', formula=roc_auc_score)
        self.metric_list = [acc, f1, roc_auc]

    def __repr__(self):
        return f"ClassifierEval <model_type={self.model_type}, apply_on={self.apply_on}," \
               f"metrics={', '.join(m.name for m in self.metric_list)}>"
