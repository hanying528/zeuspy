from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .utils.model_utils import catch_error, confusion_matrix_plot, scatter_plot_ys, timer


VALID_MODEL_TYPES = ('regression', 'classification')
VALID_ALGORITHMS = {'regression': ('Linear Regression',
                                   'SGD',
                                   'Random Forest',
                                   'XGBoost'),
                    'classification': ('Logistic Regression',
                                       'SVM',
                                       'Random Forest',
                                       'XGBoost')}


@catch_error('linear regression')
def _linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


@catch_error('logistic regression')
def _logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


class Model:
    """The core class for training and testing models"""

    def __init__(self, model_type: str, target_var: str, algorithm: str = None, model_path: str = None) -> None:
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

    def _load_model_object(self, model_path: str):
        """Load the scikit-learn model from the given path"""
        raise NotImplementedError

    def _validate_model_options(self):
        """Ensure the model type and algorithm are supported"""
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Model type must be one of the following: {VALID_MODEL_TYPES}")
        elif not self.model and not self.algorithm:
            raise ValueError("Algorithm is required if not using an existing model")
        elif self.algorithm not in VALID_ALGORITHMS[self.model_type]:
            raise NotImplementedError(f'{self.algorithm} is currently not supported.')

    def _train_test_split(self, data: pd.DataFrame,
                          ratio: float = 0.7, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                       pd.Series, pd.Series]:
        """Split data into training and testing sets based on the ratio"""
        X = data.loc[:, data.columns != self.target_var]
        y = data[self.target_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=seed)
        print(f"Splitting data into training set (shape: {X_train.shape}) and testing set (shape: {X_test.shape})")
        return X_train, X_test, y_train, y_test

    @timer("Training model took total:")
    def _train_regression_model(self, algo_name: str, X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.array, np.ndarray, pd.Series]):
        if algo_name == 'Linear Regression':
            self.model = _linear_regression(X, y)
        else:
            raise NotImplementedError

    @timer("Training model took total:")
    def _train_classifier(self, algo_name: str, X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.array, np.ndarray, pd.Series]):
        if algo_name == 'Logistic Regression':
            self.model = _logistic_regression(X, y)
        else:
            raise NotImplementedError

    def evaluate_model(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.array, np.ndarray, pd.Series],
                       apply_on: str) -> None:
        """Evaluate the model performance on the given data set"""
        y_pred = self.model.predict(X)
        if self.model_type == 'regression':
            evaluator = RegressionModelEval(apply_on=apply_on)
        else:
            evaluator = ClassifierEval(apply_on=apply_on)
        evaluator.evaluate_and_display(y, y_pred)

    def save_model(self, save_to: str = None) -> None:
        """Save model object to disk"""
        if save_to:
            # TODO: Validate the provided path
            joblib.dump(self.model, save_to)
            print(f"Saved model object to {save_to}")

    def train(self, data: pd.DataFrame, save_to: str = None) -> None:
        """Train the model on the training set and evaluate using the testing set"""
        X_train, X_test, y_train, y_test = self._train_test_split(data)
        if self.model_type == 'regression':
            self._train_regression_model(self.algorithm, X_train, y_train)

        else:
            self._train_classifier(self.algorithm, X_train, y_train)

        # Evaluate model performance on training and testing set separately
        self.evaluate_model(X_train, y_train, 'training')
        self.evaluate_model(X_test, y_test, 'testing')

        # TODO: Once parameter tuning feature is implemented, we should use the optimal
        #  parameters to re-train the model using whole dataset. For now, only export the
        #  model trained with the training set.
        # self._train_classifier(self.algorithm, X, y_train)

        if not save_to:
            suffix = datetime.today().strftime('%Y%m%d')
            save_to = f'{self.model_type}_{self.target_var}_{suffix}.pkl'
        self.save_model(save_to)


@dataclass
class Metric:
    name: str
    value: float = np.inf
    formula: Callable = None

    def calculate(self, y, y_pred):
        self.value = self.formula(y, y_pred)


class Evaluator(ABC):

    def __init__(self, apply_on: str) -> None:
        self.apply_on = apply_on
        self.metric_list = []
        self.y_actual = None
        self.y_pred = None

    @abstractmethod
    def __repr__(self):
        return

    @abstractmethod
    def _plot(self):
        return

    def _calc_metrics(self) -> None:
        if self.y_actual is None or self.y_pred is None:
            raise ValueError("Actual values and predictions must be defined before calculating evaluation metrics.")
        for metric in self.metric_list:
            metric.calculate(self.y_actual, self.y_pred)

    def evaluate_and_display(self, y: Union[list, np.array, np.ndarray, pd.Series],
                             y_pred: Union[list, np.array, np.ndarray, pd.Series]):
        if len(y) != len(y_pred):
            raise ValueError("y must have same length as y_pred")
        self.y_actual = y
        self.y_pred = y_pred
        self._calc_metrics()
        print("#" * 10 + f" Model Results ({self.apply_on} set) " + "#" * 10)
        for metric in self.metric_list:
            print(f'{metric.name}: {metric.value:.4f}')
        self._plot()


class RegressionModelEval(Evaluator):

    def __init__(self, apply_on: str) -> None:
        super().__init__(apply_on)
        self.model_type = 'regression'
        mse = Metric('Mean Squared Error', formula=mean_squared_error)
        r2 = Metric('R squared', formula=r2_score)
        self.metric_list = [mse, r2]

    def __repr__(self):
        return f"RegressionModelEval <model_type={self.model_type}, apply_on={self.apply_on}," \
               f"metrics={', '.join(m.name for m in self.metric_list)}>"

    def _plot(self):
        """Scatter plot of predicted y vs. actual y"""
        scatter_plot_ys(self.y_actual, self.y_pred, self.apply_on)


class ClassifierEval(Evaluator):

    def __init__(self, apply_on: str) -> None:
        super().__init__(apply_on)
        self.model_type = 'classification'
        acc = Metric('Accuracy', formula=accuracy_score)
        f1 = Metric('F1 score', formula=f1_score)
        roc_auc = Metric('ROC AUC', formula=roc_auc_score)
        self.metric_list = [acc, f1, roc_auc]

    def __repr__(self):
        return f"ClassifierEval <model_type={self.model_type}, apply_on={self.apply_on}," \
               f"metrics={', '.join(m.name for m in self.metric_list)}>"

    def _plot(self):
        """Plot confusion matrix created based on actual y and predicted y"""
        confusion_matrix_plot(self.y_actual, self.y_pred, self.apply_on)
