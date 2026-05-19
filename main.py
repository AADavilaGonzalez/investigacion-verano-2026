import pandas as pd
import sympy as sp
from pysr import PySRRegressor

df = pd.read_csv("dataset.csv")
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "/"],
    unary_operators=["sin", "cos", "tan"],
    model_selection="best",
    loss="loss(prediction, target) = (prediction-target)^2",
    constraints={
        "sin": 1,
        "cos": 1,
        "tan": 1
    },
    complexity_of_constants=2,
    maxdepth=5,
)
X = df[['x', 'y', 'z']].to_numpy()
y = df['w'].to_numpy()
model.fit(X, y, variable_names=["x", "y", "z"])
func = sp.simplify(model.sympy()) # pyright: ignore
print(func)
