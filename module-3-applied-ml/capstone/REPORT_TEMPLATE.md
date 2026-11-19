# <Project title>

*Author, date, one-sentence summary of the result, including its uncertainty.*

## 1. Question and success criterion
What are you predicting, for whom, and what number would count as success? (Copy from the M0 pitch.
If it changed, say how and why.)

## 2. Data
Source, size, license, columns, how labels were obtained, known problems. How the train / validation / test
split was made and when the test set was locked.

## 3. Baselines
| Model | Validation score |
|---|---|
| Trivial baseline | |
| Linear / logistic | |

## 4. Models and tuning
What you tried, the tuning procedure (search space, number of trials, selection on validation only), learning curves.

| Model | Validation (mean ± std over seeds/folds) |
|---|---|
| | |

## 5. Final evaluation
One table: the chosen model against the baselines on the **test set**, evaluated once.
Error analysis: what do the ~30 worst mistakes have in common?

## 6. What didn't work
At least three things you tried that failed or didn't matter, and what you learned from each.

## 7. Limitations and next steps
Where would this model fail? What would you do with another two weeks?

## Appendix: red-team
1. Three ways the result could be wrong, and the check you did for each.
2. Leakage audit: every place information could have crossed from validation/test into training.
3. Noise vs improvement: how big is the gain compared with the seed-to-seed spread?
4. What a reviewer found, and what you changed.
