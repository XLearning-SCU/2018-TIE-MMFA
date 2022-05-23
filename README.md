

Python implementation for [Multiple Marginal Fisher Analysis (TIE)](https://ieeexplore.ieee.org/abstract/document/8476585/).

# Introduction

MMFA is a supervised subspace learning method. Unlike the most existing methods, MMFA can automatically estimate the feature dimension and obtain the low-dimensional representation. 

### Affinity Graph Construction
<img src="https://github.com/XLearning-SCU/2018-TIE-MMFA/blob/main/graph_construction.png"  width="788" height="336" />

# Requirements

- Python 3.7
- numpy
- scikit-learn

# Datasets

Here we provide two used datasets in our experiments: AR face images and Extend Yale B face image. We resize the AR images to 55x40 and Yale images to 54x48 size.


# Training and Evaluation

```
python run.py
```

This should give the classification accuracy results on the AR and Yale datasets.

Or you can simply use MMFA as a python module and perform it on the custom data:

```
import numpy as np
import mmfa

data, labels = load_data()

# specify k_1, k_2, binary_weight
mapping = mmfa.MMFA(data, labels, k_1, k_2, binary_weight)

low_dimensional_data = np.dot(data, mapping)

# do something with the processed data
...

```

# Citation

If MMFA is useful for your research, please cite the following paper:
```
@article{huang2018mmfa,
  title = {Multiple Marginal Fisher Analysis},
  author = {Huang, Zhenyu and Zhu, Hongyuan and Zhou, Joey Tianyi and Peng, Xi},
  journal = {IEEE Transactions on Industrial Electronics},
  year = {2018},
  issn = {0278-0046},
  month = dec,
  volume = {66},
  number = {12},
  pages = {9798-9807},
  publisher = {IEEE},
  doi = {10.1109/TIE.2018.2870413},
  html = {https://ieeexplore.ieee.org/document/8476585},
  abbr = {TIE},
  bibtex_show = {true},
  keywords = {Dimensionality reduction;Learning systems;Manifolds;Task analysis;Robustness;Gaussian distribution;Estimation;Automatic dimension reduction;graph embedding;manifold learning;supervised subspace learning}
}
```

# License

MIT License
