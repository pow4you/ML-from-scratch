# %% setup
import math  # noqa: F401

from autograd import MLP, Value  # noqa: F401

# %% Step 1-3: f = a*b + c, forward and backward


# %% Step 4: new operations


# %% Step 5: numerical check, then break += on purpose


# %% Step 6: a network from Values, trained on XOR
xor_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_y = [-1.0, 1.0, 1.0, -1.0]  # tanh output, so labels are -1 / +1

# %% Step 7: counting nodes, zero_grad
