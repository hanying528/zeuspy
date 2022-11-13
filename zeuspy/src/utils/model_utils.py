import time
from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ModelTrainingError(Exception):
    def __init__(self, model_name, orig_error_msg):
        self.model_name = model_name
        self.orig_error_msg = orig_error_msg

    def __str__(self):
        return f"Error occurred when training the {self.model_name} model: {self.orig_error_msg}"


def catch_error(model_name: str):
    """A decorator that capture unknown error occurred during model training process"""
    def decorator_catch_error(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise ModelTrainingError({'model_name': model_name, 'orig_error_msg': e})
        return wrapper
    return decorator_catch_error


def timer(msg: str):
    """A simple decorator that time the function execution time"""
    def decorator_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            s = time.time()
            _  = func(*args, **kwargs)
            e = time.time()
            run_time = e - s
            if msg:
                print(f"{msg} {run_time:.2f} secs")
            else:
                print(f"Running {func.__name__!r} took total: {run_time:.2f} secs")
            return _
        return wrapper
    return decorator_timer


def scatter_plot_ys(y_true, y_pred, apply_on=''):
    color = 'b'
    marker = 'o'
    if apply_on == 'testing':
        color = 'g'
        marker = 's'
    ax = sns.regplot(x=y_pred, y=y_true, color=color, marker=marker,
                     scatter_kws=dict(s=8, alpha=0.8))

    title = 'Predicted vs. Actual\n' if not apply_on else f'Predicted vs. Actual ({apply_on})\n'
    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    plt.show()


def confusion_matrix_plot(y_true, y_pred, apply_on=''):
    cf_matrix = confusion_matrix(y_true, y_pred)
    cmap = 'Blues'
    if apply_on == 'testing':
        cmap = 'Greens'
    ax = sns.heatmap(cf_matrix, annot=True, cmap=cmap)
    title = 'Confusion Matrix\n' if not apply_on else f'Confusion Matrix ({apply_on})\n'
    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    plt.show()
