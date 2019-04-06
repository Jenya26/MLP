import numpy as np

from assertion import fail


def float_compare(expected, actual, eps=1e-6):
    return abs(expected - actual) < eps


def list_compare_arr(expected, actual, eps=1e-6):
    if len(expected) != len(actual):
        return False
    result = True
    for i in range(len(expected)):
        result = result and _equals(expected[i], actual[i], eps)
    return result


float_handlers = dict()
float_handlers[np.float64] = float_compare

list_handlers = dict()
list_handlers[np.ndarray] = list_compare_arr

handlers = dict()
handlers[float] = float_handlers
handlers[list] = list_handlers


def _equals(expected, actual, eps=1e-6):
    expected_type = type(expected)
    if expected_type not in handlers:
        fail(f"Actual handlers for '{expected_type}' not found")
    actual_handlers = handlers[expected_type]
    actual_type = type(actual)
    if actual_type not in actual_handlers:
        fail(f"Handler for '{actual_type}' not found")
    return actual_handlers[actual_type](expected, actual, eps)


def equals(expected, actual, eps=1e-6, message="Objects doesn't equals"):
    if not _equals(expected, actual, eps):
        fail(f"{message}\nExpected: {expected}\nActual: {actual}")
