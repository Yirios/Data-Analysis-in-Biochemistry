from matplotlib.pylab import plt
import numpy as np
import pandas as pd 
from scipy.stats import chi2


df = 10
X = np.linspace(chi2.ppf(0.01, df),chi2.ppf(0.99, df), 100)
Y = chi2.cdf(X, df)
print(Y)
fig,ax = plt.subplots()
ax.plot(X,Y)
plt.show()
