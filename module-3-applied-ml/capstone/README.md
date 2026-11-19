# Capstone: an end-to-end ML project

**Goal:** do one complete project the way it's done when nobody is checking: a clear question, honest
data handling, baselines first, a fair evaluation, and a write-up that says what *didn't* work.
Everything from the three modules gets used.

There are no tests and no steps here. The milestones below are the structure; the decisions are yours.
Work in `capstone/<your-project-name>/` and copy `REPORT_TEMPLATE.md` there.

## Pick a project (or invent one)

**A. Your own data.** Any project of yours that produces something you can label or measure: outputs of an
image generator, plotter drawings, sensor logs, your own listening history. You need at least 500 examples
(generate or label them yourself) and a target you can define precisely. Example questions: "can a model tell a
good generated image from a bad one by my own rating?", "can I predict how long a plot will take from the image?"

**B. Signals in a time series.** Take a free price series and predict whether the next-N-day return is positive.
Rules that make this honest: split **by time** (train on the past, validate on the next block, test on the last block),
never shuffle, build features only from information available *at the time*, include transaction costs, and compare
against "always predict up" and "predict the last move". Expect noise: most results here turn out to be
noise, and learning to *tell* is the point. This is a study in evaluation, not financial advice.

**C. A public dataset.** From UCI, OpenML or Kaggle: at least 2000 rows, a clear target, and a problem you
care about. Avoid the textbook classics (iris, titanic, digits); you have seen the tricks.

## Milestones

| | Deliverable | Done when |
|---|---|---|
| **M0** | Half-page pitch | Question, data source, target, metric, and **what result would count as success, written down before any modelling** |
| **M1** | Data card + EDA + `clean()` | You can state source, size, columns, license, known problems. The **test set is split off and locked** here. |
| **M2** | Baselines | A trivial baseline and a linear/logistic model, scored on validation, with the numbers in the report |
| **M3** | Models | At least one classical model (tree, forest, k-NN) and one neural network (PyTorch or your own numpy), tuned by a *documented* procedure on validation; learning curves for the net |
| **M4** | Honest evaluation | Test set touched **once**; uncertainty from several seeds or folds; error analysis of the ~30 worst mistakes; threshold choice if relevant |
| **M5** | Write-up (≤ 2 pages) | Fill in `REPORT_TEMPLATE.md`, including the red-team appendix |

## Rules of the road

1. Fix your metric and success criterion in M0. Changing them afterwards is allowed, but it must be written down, with the reason.
2. Everything that learns from data (scalers, imputers, selected features, thresholds) is fit on training data only.
3. One command reproduces your headline number (`python run.py` or a notebook that runs top to bottom), with seeds set and dependencies listed.
4. Keep a `LOG.md` of experiments: what you tried, the number, what you concluded. Negative results count.

## Self-review (the red-team pass)

Before you call it done, answer these in the appendix of your report:

- What are three ways this result could be *wrong* or *misleading*? What did you do to check each one?
- If the score were suspiciously high, what would you look at first? Did you look?
- How big is the improvement over the baseline compared with the run-to-run noise?
- What would break if this model met data from next month?
- Which decision in the project would you most like a second opinion on?

Then get that second opinion: send the repo to a friend or ask Claude to *red-team the report*. Fix what they find.

## Stretch goals (after the milestones)

- Re-implement the key part of your best model from scratch and match its outputs.
- Compare against a pretrained model (transfer learning) and discuss when that is and isn't worth it.
- Write a short post for someone who is where you were when you started this course.
