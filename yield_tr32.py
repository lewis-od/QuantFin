from quantfin.bonds import yield_to_maturity
import numpy as np

# Parameters for TR32 bonds
M = 133.9 # Market price (as of May 2019)
P = 100.0 # Principal
C = 4.35 / 2 # Coupon
# T = (32 - 19) # Years until maturity
T = 13
t = 0  # Current time
m = 0.5 # Coupon payments every 6 months

y = yield_to_maturity(M, P, C, T, t, m)

print("y = {0:.2%}".format(y))

# Plot yield curve
Ts = np.arange(5, 25)
ys = np.array([yield_to_maturity(M, P, C, Ti, t, m) for Ti in Ts])

# Only plot positive yields
Ts = Ts[ys > 0]
ys = ys[ys > 0]

import matplotlib.pyplot as plt
plt.plot(Ts, ys * 100)
plt.xlabel("Time to maturity (years)")
plt.ylabel("Yield (%)")
plt.title("Yield curve for TR32")
plt.show()
