""" informal module to test
"""
import dimarray as da
import matplotlib.pyplot as plt
import numpy as np

da.zeros(shape=(3,4)).plot()
a = da.array(np.random.rand(4,5))

plt.figure()
a.pcolor()
plt.figure()
a.contourf()
plt.show()

