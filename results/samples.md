# PyGPT samples (Appendix B prompts)

## Prompt A

```python
def fibonacci(n):
    """Return the nth Fibonacci number."""

def fibonacci_get_first_name(n):
    """Return the first Fibonacci number."""

```

```python
def fibonacci(n):
    """Return the nth Fibonacci number."""
	return fibonacci_1_0[n]

```

## Prompt B

```python
import numpy as np

def normalize(arr):
    """Scale arr to zero mean and unit variance."""
	if arr.ndim == 2:
		return arr
	elif arr.ndim == 3:
		return arr * arr
	else:
		raise ValueError("Unsupported axis for arr")

```

```python
import numpy as np

def normalize(arr):
    """Scale arr to zero mean and unit variance."""
#    return np.median(arr, axis=-1)
#    return np.mean(arr, axis=-1)
#    return np.std(arr, axis=-1)
#    return np.std(arr, axis=-1)
#    return arr
#   
#   
def get_data_names(arr):
    """Get data names from array."""
    return dict(arr.keys())
#   

```

## Prompt C

```python
class Stack:
    def __init__(self):
	self.stack = []
    def append(self, p):
        self.stack.append(p)
    def remove(self, p):
        self.stack.remove(p)
    def __str__(self):
        return "Stack: " + str(self.stack)

```

```python
class Stack:
    def __init__(self):
	self.stack = []
	self.stack.append("Stack")
	self.stack.append("Stack.Stack")
	self.stack.append("Stack.Stack.Name")
	self.stack.append("Stack.Stack.State")
	self.stack.append("Stack.Stack.Name")
	self.stack.append("Stack.Stack.State.StackName")
```

## Prompt D

```python
def is_prime(n):
	if n < 0:
		return False
	if n < 0:
		return True
	if n < 0:
		return False
	return True

```

```python
def is_prime(n):
	return n <= 1 or n == 0 or n == 1

```

## Prompt E

```python
def read_json(path):
    """Load a JSON file and return the parsed object."""
	yaml_data = open(path, "r").read()
	return yaml_data

```

```python
def read_json(path):
    """Load a JSON file and return the parsed object."""
	with open(path, "r") as f:
	    return json.load(f)

```

## Prompt F

```python
def solve_sudoku(board):
	for i in range(len(board)):
		for j in range(len(board[i])):
			if board[i][j] == "Yes" and board[i][j] == "No":
				board[i][j] = "Yes"
			else:
				board[i][j] = "No"

```

```python
def solve_sudoku(board):
	"""
	Ensures the Lights of a board.
	"""
	try:
		board.lights.lights.lights_per_group = board.lights.lights_per_group
	except AttributeError:
		board.lights.lights.lights_per_group = 1

	l = len(board.lights)
	while True:
		try:
```
