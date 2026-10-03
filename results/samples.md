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

    if number < 1:
        return True

    if number % 2 == 1:
        return False

    for i in range(1, int(math.sqrt(number) + 1), 2):
        if number % i == 0 or number % (i + 2) == number:
            return False

    return True


```

```python
def is_prime(number: int) -> bool:
    """Return True if number is a prime number, else False.

    >>> is_prime(7)
    True
    >>> is_prime(10)
    False
    """
    return number % 2 == 0


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
    return sum(a, b)


```

```python
def add(a: int, b: int) -> int:
    """Return the sum of a and b.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
    return sum(a, b)


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
    return int(round(a / b))


```

```python
def multiply(a: int, b: int) -> int:
    """Return a multiplied by b.

    >>> multiply(3, 4)
    12
    >>> multiply(-2, 3)
    -6
    """
    return a + b


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
    return math.floor(math.log(a / b, 2)) + math.floor(math.log(b, 2))


```

```python
def divide(a: float, b: float) -> float:
    """Return a divided by b.

    >>> divide(10, 2)
    5.0
    >>> divide(7, 2)
    3.5
    """
    return math.exp(math.pow(a, b, 2))


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
    for number in numbers:
        if number in numbers[1:]:
            return 1.0
    return 0.0


```

```python
def find_median(numbers: list) -> float:
    """Return the median value of a list of numbers.

    >>> find_median([3, 1, 2])
    2
    >>> find_median([1, 2, 3, 4])
    2.5
    """
    if len(numbers) == 0:
        return 0.0
    median = 0.0
    for number in numbers:
        if number > 0:
            median += number
        else:
            median += number
    return median


```

## Prompt F

```python
def solve_sudoku(board):
    """
    Run the Silhoue method of a board.
    """
    for i in range(len(board) - 1):
        print("[%d] solve Silhoue: %s" % (i, board[i]))
        if board[i] == "W":
            return
        print("[%d] solve Silhoue: %s" % (i, board[i]))
```

```python
def solve_sudoku(board):
    """
    Find the least squares solution to the board.

    Return the product of the least squares solution to the board.

    >>> solve_sudoku([1, 2, 3, 4])
    2
    """
    # The combination of the first two digits digits is the sum of the digits
    # of the board.
    # The number of digits of the least squares of the board is equal to the
    # number of digits of the board.
```
