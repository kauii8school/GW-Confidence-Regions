import math 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
from GW import *

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "Main", "Images"))


delta = 100
thetaList = np.linspace(-math.pi/2, math.pi/2, delta)
phiList = np.linspace(-math.pi, math.pi, delta)

psi = 0
Z = np.zeros((delta, delta))
for i, phi in enumerate(thetaList):
    for j, theta in enumerate(phiList):
        z = earthDetectorNetwork.getAntennaPowerPattern(theta, phi, psi)
        Z[i, j] = z

fig, ax = plt.subplots()
CS = ax.contour(phiList, thetaList, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")
ax.set_title("Antenna Power for LIGO and VIRGO")
plt.show() 