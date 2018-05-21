# Coding guidlines (suggestion)

## Variables

### Naming conventions
Speaking variable names help others to understand your code:

```python
input_idx       # if it is actually an index
learnrate       # or learn_rate
total_loss
mean_days
is_done = True  # For boolean variables
```

instead of

```python
input
alpha
sum
agv_d
flag
```

exceptions can be variables in not deeply nested loops

```python
for k, v in dict:
```

### Instantiation

No multiple one line initialisations because it is hard to get all variables in one sight
```python
arr1, ste2, hello2, v_ge = [], [], [], []
```

better to do each one in one line:
```python
frames = []
rewards = []
weather_today = []
```

Try to use as less variables as possible and use good naming convention
to avoid having several copys of the same variable.

Not to:

```python
x = 10
...
num_feat = x
...
input = num_feat
```
better:

```python
input_features = 10
```

It increases readability and reduces error-prone constructions. E.g. on a numpy array
several assignments create shallow copies. So a change on the second variable affects
unexpectedly the first variable.

```python
x = np.array([1, 2])
...
features = x
features[0] = 99
print(x)  # returns [99, 2]
```

## Functions

If too many function arguments for one libe then align where the first argument starts.
This makes ot easier to differentiate between the arguments and the function content.

```python
def my_function(arg1,
                arg2,
                arg3):
    # Whatever comes here
```

Use speaking names for functions, i.e. it should include a verb.

```python
def get_frames():

def set_frames():

def is_avaliable():
    returns True

def draw():
```

Insert an empty line between two function declarations

```python
def do_this():
    return "Hi"

def do_that():
    return "Hello"
```

Instead of

```python
def do_this():
    return "Hi"
def do_that():
    return "Hello"
```

# General stuff

Python 3 code guidlines include also

### Use double quotes
```python
my_string = "hello"
```

### Spacing
Leave space between operators
```python
total_days = 2 * (days + 1)
```

instead of
```python
total_days=2*(days+1)
```

### Avoid magic numbers

Not naming numbers is very confusing for others and lead to
decisions which might not be understood after a while

```python
arrays = np.array([1, 2], 1, 4, True, 736)

new_mean = mean * 0.01
```

better is

```python
arrays = np.array([1, 2], axis=1, size=4, inplace=True, repeat=736)
```

```python
steplenght = 0.01
adjusted_mean = mean * steplenght
```
