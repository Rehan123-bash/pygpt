# PyGPT samples (Appendix B prompts)

## Prompt A

```python
def is_prime(number: int) -> bool:
    """Return True if number is a prime number, else False.

    >>> is_prime(7)
    True
    >>> is_prime(10)
    False
    """
    return int(number) % 2 == 0


```

```python
def is_prime(number: int) -> bool:
    """Return True if number is a prime number, else False.

    >>> is_prime(7)
    True
    >>> is_prime(10)
    False
    """

    if not isinstance(number, int) or number < 0:
        return False

    return number % 2 == 1


```

## Prompt B

```python
def add(a: int, b: int) -> int:
    """Return the sum of a and b.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
    return a + b


```

```python
def add(a: int, b: int) -> int:
    """Return the sum of a and b.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
    return a + b


```

## Prompt C

```python
def multiply(a: int, b: int) -> int:
    """Return a multiplied by b.

    >>> multiply(3, 4)
    12
    >>> multiply(-2, 3)
    -6
    """
    return a * b


```

```python
def multiply(a: int, b: int) -> int:
    """Return a multiplied by b.

    >>> multiply(3, 4)
    12
    >>> multiply(-2, 3)
    -6
    """
    return a * b


```

## Prompt D

```python
def divide(a: float, b: float) -> float:
    """Return a divided by b.

    >>> divide(10, 2)
    5.0
    >>> divide(7, 2)
    3.5
    """
    return a / b


```

```python
def divide(a: float, b: float) -> float:
    """Return a divided by b.

    >>> divide(10, 2)
    5.0
    >>> divide(7, 2)
    3.5
    """
    return a / b


```

## Prompt E

```python
def find_median(numbers: list) -> float:
    """Return the median value of a list of numbers.

    >>> find_median([3, 1, 2])
    2
    >>> find_median([1, 2, 3, 4])
    2.5
    """
    median = 0
    for number in numbers:
        if number > median:
            median = number
            median = median + 1
    return median


```

```python
def find_median(numbers: list) -> float:
    """Return the median value of a list of numbers.

    >>> find_median([3, 1, 2])
    2
    >>> find_median([1, 2, 3, 4])
    2.5
    """
    return sum(numbers) / len(numbers)


```

## Prompt F

```python
def solve_sudoku(board):
    return [round(number * (1 - number) * (1 - number) * (number - 1) * (number - 1) for number in range(1, 7)] for number in range(1, 7) ]

```

```python
def solve_sudoku(board):
    """
    Solve the Sudoku board using the Sudoku algorithm.
    """
    return solve(board, lambda x: x[1] - x[0], lambda x: x[0] - x[1])

```
