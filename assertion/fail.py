from .assertion_fail import AssertionFail

__all__ = ['fail']


def fail(message="Assertion fail"):
    raise AssertionFail(message)
