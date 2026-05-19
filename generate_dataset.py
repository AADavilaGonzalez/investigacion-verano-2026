import sympy as sp
import numpy as np
import pandas as pd

FUNCTION_DECLARATION = "w(x, y, z)"
FUNCTION_DEFINITION = "4*x + 7*y + 3*z"
VARIABLE_BOUNDS = [(0,10), (0,10), (0,10)]
DATASET_SIZE = 500
LOG_ONLY = False

name, _, body = FUNCTION_DECLARATION.partition("(")
name = name.strip()
body = body[:-1]
variable_names = [name.strip() for name in body.split(',')]
symbols = sp.symbols(variable_names)
if len(symbols) != len(VARIABLE_BOUNDS):
    raise ValueError(
        "The length of the variable bounds array does not match the amount of variables")

expr = sp.parse_expr(FUNCTION_DEFINITION)
if set(symbols) != expr.free_symbols:
    raise ValueError(
        "The variables used between function definiton and declaration do not match")
func = sp.lambdify(symbols, expr, 'numpy')

inputs = [
    np.random.uniform(low, high, DATASET_SIZE)
    for low, high in VARIABLE_BOUNDS
]
outputs = func(*inputs)

data = {
    variable_name: variable_data
    for variable_name, variable_data
    in zip(variable_names, inputs)
}
data[name] = outputs

df = pd.DataFrame(data)
if LOG_ONLY:
    print(df.head(10))
else:
    df.to_csv("dataset.csv", index=False)
