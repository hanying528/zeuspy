# zeuspy
A machine learning GUI in Jupyter

## Dev Environment Setup

```bash
conda create -n zeuspy python=3.9 -y
conda activate zeuspy
pip install -e .
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

### Create dummy dataset for testing

#### 1. Regression

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


x, y = make_regression(n_samples=1000,
                       n_features=8,
                       noise=80,
                       random_state=42)
# save x, y to disk
arrs = np.concatenate((x, y[:, None]), axis=1)
df = pd.DataFrame(data=arrs, columns=[f'feature{i}' for i in range(8)] + ['y'])
df.to_csv('data/toy_regression.csv', index=False)
```

#### 2. Classification
```python
from sklearn.datasets import make_classification


x, y = make_classification(n_samples=1000,
                           n_features=8,
                           n_informative=2,
                           n_clusters_per_class=1,
                           random_state=42)
# save x, y to disk
arrs = np.concatenate((x, y[:, None]), axis=1)
df = pd.DataFrame(data=arrs, columns=[f'feature{i}' for i in range(8)] + ['y'])
df.to_csv('data/toy_classification.csv', index=False)
```