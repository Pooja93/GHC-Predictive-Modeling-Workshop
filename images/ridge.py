print(__doc__)

# Author: Mathieu Blondel
#         Jake Vanderplas
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression 
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge

numpoints=50;

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * (np.sin(x)+0.3*np.random.normal(0,1,len(x)));


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:numpoints])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
#plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
#         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

degree=8
count=0;
model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
model.fit(X, y)
y_plot = model.predict(X_plot)
plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="Linear degree %d" % degree)
count=count+1;
model = KernelRidge(alpha=.01,kernel="poly",degree=degree)
#model = make_pipeline(PolynomialFeatures(degree), linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]))
model.fit(X, y)
y_plot = model.predict(X_plot)
plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="Kernel Ridge degree %d" % degree)

#for count, degree in enumerate([3, 4, 10]):
#    #model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
#    #model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=.1))
#    model = make_pipeline(PolynomialFeatures(degree), linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]))
#    model.fit(X, y)
#    y_plot = model.predict(X_plot)
#    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
#             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()
