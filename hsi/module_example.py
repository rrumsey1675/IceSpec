"""This module serves primarily as an example template. For every Python 
module, we should make an effort to use type hints 
(https://docs.python.org/3/library/typing.html), supply unit 
tests (https://docs.python.org/3/library/unittest.html), and we should include
docstrings (https://peps.python.org/pep-0257/)
"""
from typing import List


def sum_list(lst: List[int]) -> int:
    """The summation of all integers in the list.

    Args:
        lst (List[int]): Some list of non-None integers

    Returns:
        int: The sum of all integers
    """
    result = 0
    for i in lst:
        result += i
    return result  # This is stupid, why not simply sum(lst)?
