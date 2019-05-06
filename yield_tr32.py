from quantfin.bonds import yield_to_maturity
import numpy as np

# Parameters for TR32 bonds
M = 133.9 # Market price (as of May 2019)
P = 100.0 # Principal
C = 4.35 # Coupon
T = (32 - 19) # Years until maturity
t = 0  # Current time
ti = np.arange(t, T, 0.5) # Coupon payments every 6 months

y = yield_to_maturity(M, P, C, T, t, ti)

print("y = {0:.2%}".format(y))
