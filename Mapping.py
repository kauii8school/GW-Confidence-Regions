import math 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os

from GWFunctions import *

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "Main", "Images"))


delta = 100
theta = np.linspace(0, math.pi, delta)
phi = np.linspace(0, 2 * math.pi, delta)
thetaMatrix, phiMatrix = np.meshgrid(theta, phi)
print(thetaMatrix, phiMatrix)
Z = testFunction(thetaMatrix, phiMatrix)

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)

fig, ax = plt.subplots()
CS = ax.contour(theta, phi, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")
ax.set_title(r'f = sin($\theta$)sin($\phi$)')
plt.show()
print(Z)