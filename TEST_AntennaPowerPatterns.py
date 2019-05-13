import math
import numpy as np
from GW import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

delta = 100
thetaList = np.linspace(0, math.pi, delta)
phiList = np.linspace(0, 2 * math.pi, delta)
psiList = np.linspace(0, 2 * math.pi, delta)
psi = 0

maxCross, maxPlus, maxAntennaPowerPattern = 0, 0, 0
detectorToTest = Virgo
for i, theta in enumerate(thetaList):
    for j, phi in enumerate(phiList):

        a = detectorToTest.afunction(theta, phi)
        b = detectorToTest.bfunction(theta, phi)
        cross = detectorToTest.AP_CROSS(a, b, psi)
        plus = detectorToTest.AP_PLUS(a, b, psi)
        antennaPowerPattern = (cross ** 2) + (plus ** 2)

        if math.fabs(cross) > maxCross:
            maxThetaCross, maxPhiCross = theta, phi
            maxCross = math.fabs(cross)

        if math.fabs(plus) > maxPlus:
            maxThetaPlus, maxPhiPlus = theta, phi
            maxPlus = math.fabs(plus)

        if math.fabs(antennaPowerPattern) > maxAntennaPowerPattern:
            maxThetaApp, maxPhiApp = theta, phi
            maxAntennaPowerPattern = math.fabs(antennaPowerPattern)

print("plus  {}  theta {}    phi {}".format(maxPlus, maxThetaPlus, maxPhiPlus))
print("cross {}  theta {}    phi {}".format(maxCross, maxThetaCross, maxPhiCross))
print("app   {}  theta {}    phi {}".format(maxAntennaPowerPattern, maxThetaApp, maxPhiApp))


delta = 100
thetaList = np.linspace(-math.pi/2, math.pi/2, delta)
phiList = np.linspace(-math.pi, math.pi, delta)
psi = 0
ZList, plotThetaList, plotPhiList, plotXList, plotYList = ([], [], [], [], [])
for i, phi in enumerate(phiList):
    for j, theta in enumerate(thetaList):
        z = earthDetectorNetwork.getAntennaPowerPattern(theta, phi, psi)
        plotThetaList.append(theta)
        plotPhiList.append(phi)
        plotXList.append(math.sin(theta) * math.cos(phi))
        plotYList.append(math.sin(theta) * math.sin(phi))
        ZList.append(z)

# print(max(ZList))
# fig, ax = plt.subplots()
# CS = ax.tricontour(plotPhiList, plotThetaList, ZList)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_xlabel(r"X")
# ax.set_ylabel(r"Y")
# ax.set_title("Antenna Power Patterns for LIGO and VIRGO")
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
CS = ax.plot_trisurf(plotXList, plotYList, ZList)
ax.scatter(plotXList, plotYList, ZList, c='b', marker='o')
ax.plot_trisurf(plotXList, plotYList, ZList)
plt.show()