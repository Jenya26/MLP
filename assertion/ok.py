from assertion import equals

__all__ = ['ok']


def ok():
    equals(True, True)
