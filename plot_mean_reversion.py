from quantfin.random_walk import mean_reversion

x1 = mean_reversion(500, 1.0, 1.0, 1.0)
x2 = mean_reversion(500, 1.0, 1.0, 0.1)
x3 = mean_reversion(500, 1.0, 1.0, 0.01)

import matplotlib.pyplot as plt

y_max = max(x1.max(), x2.max(), x3.max())
y_min = min(x1.min(), x2.min(), x3.min())

plt.plot(x1)
plt.plot(x2)
plt.plot(x3)

plt.xlim((0, 500))
plt.ylim((y_min, y_max))

plt.legend(["a = 1.0", "a = 0.1", "a = 0.01"])

plt.show()
