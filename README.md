# Quant Fin
Code for doing various quantitative finance things. Very much work in progress.

Currently:
- [bonds.py](quantfin/bonds.py) has a function to calculate the yield to maturity of a bond   
- [random_walk.py](quantfin/random_walk.py) generates geometric Brownian or mean reverting random walks 
(for use in Monte Carlo simulations)   
- [swaps.py](quantfin/swaps.py) contains a function to calculate the fixed rate to use in a interest-rate swap
- [volatility.py](quantfin/volatility.py) contains functions to calculate volatility from returns data using various models

The files in the root directory of the repository use the above to perform various calculations and make some plots.
