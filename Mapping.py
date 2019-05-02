import math 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
from GW import *
from mpl_toolkits.mplot3d import Axes3D

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "Main", "Images"))


delta = 100
thetaList = np.linspace(-math.pi/2, math.pi/2, delta)
phiList = np.linspace(-math.pi, math.pi, delta)

psi = 0
Z = np.zeros((delta, delta))
ZList, plotThetaList, plotPhiList, plotXList, plotYList = ([], [], [], [], [])
for i, phi in enumerate(thetaList):
    for j, theta in enumerate(phiList):
        z = earthDetectorNetwork.getAntennaPowerPattern(theta, phi, psi)
        fplus = GWDetector.Single_AP_PLUS(theta, phi, psi)
        fcross = GWDetector.Single_AP_CROSS(theta, phi, psi)
        #z = (fplus ** 2) + (fcross ** 2)
        plotThetaList.append(theta)
        plotPhiList.append(phi)
        plotXList.append(math.sin(theta) * math.cos(phi))
        plotYList.append(math.sin(theta) * math.sin(phi))
        ZList.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plotXList, plotYList, ZList, c='b', marker='o')
ax.plot_trisurf(plotXList, plotYList, ZList)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

fig, ax = plt.subplots()
CS = ax.tricontour(plotThetaList, plotPhiList, ZList)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")
ax.set_title("Antenna Power for LIGO and VIRGO")
plt.show() 