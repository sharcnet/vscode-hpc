#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

from time import perf_counter

n_tosses = 1000000

#<---- start timing
start = perf_counter()

x = np.random.rand(n_tosses)
y = np.random.rand(n_tosses)

n_in_circle = 0
for i in range(0, n_tosses - 1):
    if (x[i]**2 + y[i]**2 <= 1):
        n_in_circle += 1

end = perf_counter()
#<---- end timing

pi_estimate = 4 * (n_in_circle / n_tosses)
print("Monte-Carlo Pi Estimator")
print("Estimated π: ", pi_estimate)
print("Percent error: ", 100 * math.fabs(math.pi - pi_estimate) / math.pi, "%")
print("Elapsed time: ", end - start, "seconds")

circle_x = x[np.sqrt(x**2 + y**2) <= 1]
circle_y = y[np.sqrt(x**2 + y**2) <= 1]

fig = plt.figure()
plot = fig.add_subplot(111)
plot.scatter(x, y, marker='.', color='blue')
plot.scatter(circle_x, circle_y, marker='.', color='red')

x = np.linspace(0, 1, 100)
y = np.sqrt(1 - x**2)

plot.plot(x, y, color='black')

plot.set_aspect(1.0)

plt.rcParams['figure.figsize'] = [20, 20]
plt.show()