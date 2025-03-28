# Deep Learning Challenge: Alphabet Soup Charity

## Overview

The goal here was to build a neural network model that could predict whether funding applicants for Alphabet Soup Charity would be successful. I used TensorFlow and Keras to handle the modeling, and ran multiple optimization attempts to push performance past the 75% threshold.

---

## Files Included

- `notebooks/AlphabetSoupCharity.ipynb` — baseline model
- `notebooks/AlphabetSoupCharity_Optimization1.ipynb` — added a third hidden layer
- `notebooks/AlphabetSoupCharity_Optimization2.ipynb` — increased neuron count in the first layer
- `notebooks/AlphabetSoupCharity_Optimization3.ipynb` — swapped activation to `tanh`
- `h5/` — contains the exported `.h5` model files from each version
- `NNN_Report.docx` — written breakdown of the modeling process

---

## Tools + Libraries

- Python 3
- pandas, numpy
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

---

## Modeling Process

- Scaled the input features with `StandardScaler`
- Categorical variables were encoded using `get_dummies()`
- The model used binary classification with `sigmoid` output and `binary_crossentropy` loss
- I started with a basic 2-hidden-layer network and made three optimization passes:
  - **Attempt 1**: Added a third hidden layer
  - **Attempt 2**: Boosted neuron count to 128-64-1
  - **Attempt 3**: Switched activation functions to `tanh`

---

## Results

| Model Version          | Accuracy   |
|------------------------|------------|
| Baseline               | 68.66%     |
| Optimization Attempt 1 | 73.32%     |
| Optimization Attempt 2 | 62.65%     |
| Optimization Attempt 3 | **69.80%** ✅ |

Attempt 3 ended up performing the best, even though it didn’t break 75%. It kept the same layer structure as the baseline but added small adjustments that helped squeeze out better accuracy. Final model was saved as `AlphabetSoupCharity_Optimization3.h5`.

---

## Final Thoughts

Breaking past 70% accuracy wasn’t too bad, but pushing beyond that was rough. More layers gave a boost early on, but throwing in extra neurons backfired. What actually moved the needle was switching up the activation function. If I were to keep going, I’d test different optimizers, experiment with dropout, or try to address the class imbalance in the dataset.
