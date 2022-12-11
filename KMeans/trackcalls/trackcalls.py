import functools


def trackcalls(func):
    """
    The goal of this function is to record
    wheter or not it was used before.

    Arguments:
        func: The function to be tracked

    Returns:
        wrapper: boolean: True if the function
        was used, else False"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False
    return wrapper
