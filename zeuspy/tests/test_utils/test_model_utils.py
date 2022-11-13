from pytest import fixture

from zeuspy.src.utils.model_utils import ModelTrainingError, catch_error


@fixture
def dummy_mte():
    return ModelTrainingError('Foo', 'I am the original error message')


@catch_error('ModelX')
def bar():
    raise AssertionError('1 != 2')


def test_mte_init(dummy_mte):
    assert dummy_mte.model_name == 'Foo'
    assert dummy_mte.orig_error_msg == 'I am the original error message'


def test_mte_string_representation(dummy_mte):
    assert str(dummy_mte) == "Error occurred when training the Foo model: I am the original error message"


def test_catch_error_works():
    try:
        bar()
    except ModelTrainingError as e:
        assert str(e) == "Error occurred when training the ModelX model: 1 != 2"
