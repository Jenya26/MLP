import numpy as np

from .fail import fail


def types_not_comparable(expected, actual, eps=1e-6):
    return False


def bool_compare(expected, actual, eps=1e-6):
    return expected == actual


def float_compare(expected, actual, eps=1e-6):
    return abs(expected - actual) < eps


def list_compare_arr(expected, actual, eps=1e-6):
    if len(expected) != len(actual):
        return False
    result = True
    for i in range(len(expected)):
        result = result and _equals(expected[i], actual[i], eps)
    return result


bool_handlers = dict()
bool_handlers[bool] = bool_compare

float_handlers = dict()
float_handlers[np.float64] = float_compare
float_handlers[float] = float_compare
float_handlers[np.ndarray] = types_not_comparable

list_handlers = dict()
list_handlers[np.ndarray] = list_compare_arr
list_handlers[list] = list_compare_arr
list_handlers[float] = types_not_comparable

handlers = dict()
handlers[bool] = bool_handlers
handlers[float] = float_handlers
handlers[np.float64] = float_handlers
handlers[list] = list_handlers
handlers[np.ndarray] = list_handlers


def _equals(expected, actual, eps=1e-6):
    expected_type = type(expected)
    actual_type = type(actual)
    if expected_type not in handlers:
        fail(f"Handler for ({expected_type}, {actual_type}) not found")
    actual_handlers = handlers[expected_type]
    if actual_type not in actual_handlers:
        fail(f"Handler for ({expected_type}, {actual_type}) not found")
    return actual_handlers[actual_type](expected, actual, eps)


def equals(expected, actual, eps=1e-6, message="Objects doesn't equals"):
    if not _equals(expected, actual, eps):
        fail(f"{message}\nExpected: {expected}\nActual: {actual}")
