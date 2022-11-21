from functools import partial

import ipywidgets as widgets
from IPython.display import display

from .model import VALID_ALGORITHMS
from .utils.gui_utils import load_data, train_model

DEFAULT_MODEL_TYPE = 'regression'


def model_training_interface():
    """Instantiate user interface for training model"""

    style = {'description_width': '150px'}
    box_layout = widgets.Layout(margin='10px 10px 10px 20px',
                                display='flex')
    # input data
    input_data_box = widgets.Text(value='../data/toy_regression.csv',
                                  description='Input Data Location',
                                  style=style,
                                  layout=widgets.Layout(width='400px'))
    load_btn = widgets.Button(description='Load Data',
                              disabled=False,
                              button_style='primary',
                              style={'description_width': 'initial',
                                     'button_width': 'auto'},
                              icon='plus')
    # model type
    select_model_type = widgets.Dropdown(
        options=['regression', 'classification'],
        value=DEFAULT_MODEL_TYPE,
        description='Model type:',
        style=style,
        disabled=False
    )

    # capture stdout
    out = widgets.Output(layout=box_layout)

    def on_click_train_model(model_type, algorithm, target_var, *args):
        """Trigger model training process"""
        with out:
            train_model(load_data.data, model_type.value, algorithm.value, target_var.value)

    def on_click_confirm(model_type, target_var, *args):
        """Enable algorithm selection and train model button"""
        with out:
            allowed_algos = VALID_ALGORITHMS[model_type.value]
            select_algo = widgets.Dropdown(options=allowed_algos,
                                           description='Select an algorithm:',
                                           style={'description_width': '200px'},
                                           layout=widgets.Layout(width='350px'),
                                           disabled=False)
            # enable train model button
            train_model_btn = widgets.Button(description='Start training!',
                                             disabled=False,
                                             button_style='primary',
                                             style={'description_width': 'initial',
                                                    'button_width': 'auto'})
            train_model_btn.on_click(partial(on_click_train_model, model_type, select_algo, target_var))
            display(widgets.HBox([select_algo, train_model_btn],
                                 layout=box_layout))

    def on_click_load_data(input_data_field, *args):
        """Trigger data loading process and enable more widgets"""
        with out:
            load_data(input_data_field.value)
            # enable target var dropdown list
            cols = load_data.data.columns
            select_target_var = widgets.Dropdown(options=cols,
                                                 description='Select target variable:',
                                                 style={'description_width': '200px'},
                                                 disabled=False)
            # confirm button
            confirm_btn = widgets.Button(description='Confirm',
                                         button_style='success',
                                         isabled=False)
            confirm_btn.on_click(partial(on_click_confirm, select_model_type, select_target_var))
            display(widgets.HBox([select_model_type, select_target_var, confirm_btn],
                                 layout=widgets.Layout(margin='20px 10px 20px 0px', align_items='flex-start')))

    load_btn.on_click(partial(on_click_load_data, input_data_box))
    box = widgets.HBox([input_data_box, load_btn],
                       layout=box_layout)
    display(widgets.VBox([box, out]))


def model_testing_interface():
    raise NotImplementedError
