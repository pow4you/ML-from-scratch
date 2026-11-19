# ML from scratch

A hands-on course in data science, neural networks and applied machine learning.
You build the pieces yourself, so you understand what the libraries do for you.

There are no finished solutions in this repo, on purpose. Each exercise gives you
a short primer, a set of steps, and a starter file with the boring plumbing already done.

## How an exercise works

Every exercise folder contains:

| File | What it is |
|---|---|
| `README.md` | Goal, short primer, and the steps |
| `starter.py` (or a named module) | Cells to run one at a time (`# %%` works in VS Code, PyCharm and Jupytext) |
| `test_*.py` | Only where there is a well-defined right answer (gradients, shapes, etc.) |

Each exercise is split roughly **two-thirds guided, one-third challenge**:

- 🟢 **Guided** steps lead you to understand the material. Everything you need is in the primer. Experiment freely.
- 🔴 **Challenge** steps give you a goal and no recipe. This is where it sticks.

Rules of the game (they're yours to bend):

1. Spend about 20 minutes on a step before opening a hint. Hints are in `<details>` blocks.
2. Challenges have no hints. If you're stuck, simplify the problem, then simplify it again.
3. Finish an exercise by answering its **"Can you explain it?"** questions out loud or in writing. If you can't, you haven't finished yet.
4. When you're done with a step, ask for a review: "red-team this". Reviewing is where mistakes get caught.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Modules 1 and 2
pip install -r requirements-torch.txt   # Module 3 (PyTorch, CPU is fine)
python data/make_messy_data.py     # only if data/housing_messy.csv is missing
```

Run the tests for the exercise you're working on, not the whole repo.
Tests for exercises you haven't started will fail by design:

```bash
pytest module-2-neural-nets/ex01-one-neuron -q
```

## Contents

### Module 1: Data science (6 exercises)
- [ ] [01 Meet your data](module-1-data-science/ex01-meet-your-data) (4 steps)
- [ ] [02 Cleaning](module-1-data-science/ex02-cleaning) (5 steps)
- [ ] [03 Seeing the data](module-1-data-science/ex03-seeing-the-data) (4 steps)
- [ ] [04 Relationships](module-1-data-science/ex04-relationships) (5 steps)
- [ ] [05 Splits and baselines](module-1-data-science/ex05-split-and-baseline) (4 steps)
- [ ] [06 Linear regression by hand](module-1-data-science/ex06-linear-regression) (5 steps)

### Module 2: Neural net basics and training parameters (5 exercises)
- [ ] [01 One neuron learns a line](module-2-neural-nets/ex01-one-neuron) (6 steps)
- [ ] [02 Losses and activations](module-2-neural-nets/ex02-losses-and-activations) (5 steps)
- [ ] [03 Autograd from scratch](module-2-neural-nets/ex03-autograd) (7 steps)
- [ ] [04 An MLP in numpy](module-2-neural-nets/ex04-mlp-numpy) (7 steps)
- [ ] [05 Training parameters lab](module-2-neural-nets/ex05-training-lab) (6 steps)

### Module 3: Applied machine learning (4 exercises + capstone)
- [ ] [01 k-NN and trees](module-3-applied-ml/ex01-knn-and-trees) (4 steps)
- [ ] [02 PyTorch](module-3-applied-ml/ex02-pytorch) (5 steps)
- [ ] [03 Evaluation done honestly](module-3-applied-ml/ex03-evaluation) (4 steps)
- [ ] [04 Images and convolutions](module-3-applied-ml/ex04-images-and-convolutions) (5 steps)
- [ ] [Capstone project](module-3-applied-ml/capstone)

## The dependency chain

Module 1 ends with linear regression, which is the loss function that Module 2 starts
minimising with gradient descent. Module 2's autograd and MLP are what PyTorch gives
you in Module 3. Do the modules in order; within a module, the exercises build on each other.

## Rough time budget

Module 1: 1–2 hours per exercise. Module 2: 2–3 hours per exercise. Module 3: 2–3 hours per
exercise. Capstone: 10–20 hours. These are guesses. Going slower and understanding it
beats going fast.
