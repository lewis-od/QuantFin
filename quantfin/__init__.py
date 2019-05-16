"""
A python module for performing various quantitative finance calculations. Very
much work in progress.

- `quantfin.bonds` has a function to calculate the yield to maturity of a bond
- `quantfin.random_walk` generates geometric Brownian or mean reverting random
walks (for use in Monte Carlo simulations)
- `quantfin.swaps` contains a function to calculate the fixed rate to use in a
interest-rate swap
- `quantfin.volatility` contains functions to calculate volatility from returns
data using various models
"""

from . import bonds
from . import swaps
from . import random_walk
from . import volatility
from . import data
from . import utils
