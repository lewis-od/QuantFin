from quantfin.random_walk import geometric_brownian

x1 = geometric_brownian(500, 100, 0.0, 0.05)
x2 = geometric_brownian(500, 100, 0.0, 0.05)
x3 = geometric_brownian(500, 100, 0.0, 0.05)
x4 = geometric_brownian(500, 100, 0.0, 0.05)

y_min = 0.0
y_max = max(x1.max(), x2.max(), x3.max(), x4.max())

import matplotlib.pyplot as plt

plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)

plt.ylim((y_min, y_max))
plt.xlim((0, 500))

plt.show()
