import math 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
from GW import *
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.path as mppath
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "Main", "Images"))

#Event generation number of events generated is proportional to d^3 but we don't really care about the proportionality constant as it will be divided out eventually anyways
class Event:

    def __init__(self, _theta, _phi, _psi, _distance):
        self.theta = _theta
        self.phi = _phi
        self.psi = _psi
        self.X = math.sin(_theta) * math.cos(_phi)
        self.Y = math.sin(_theta) * math.sin(_phi)
        self.XYPoint = (self.X, self.Y)
        self.distance = _distance
        self.thetaPhiPoint = (self.theta, self.phi)


#Finding how many events at each distance
k = 1
maxDistance = 170
numDistancePoints = 200
distanceList = np.linspace(0, maxDistance, numDistancePoints)
numEventsAtDistance = []
for distance in distanceList:
    eventsForDistance = k * (distance ** 3)
    numEventsAtDistance.append(eventsForDistance)

approxTotalEvents = 10000
tempTotalEvents = sum(numEventsAtDistance)
for i, distance in enumerate(distanceList):
    percentEvents = numEventsAtDistance[i] / tempTotalEvents
    numEventsAtDistance[i] = int(round(percentEvents * approxTotalEvents))

trueTotalEvents = sum(numEventsAtDistance)

#Event generation
eventList = []
for i, distance in enumerate(distanceList):
    for j in range(0, numEventsAtDistance[i]):
        theta, phi, psi = random.uniform(0, math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)
        eventList.append(Event(theta, phi, psi, distance))

detectedEventList = []
for event in eventList:
    z = GWDetector.getSingleAntennaPowerPattern(event.theta, event.phi, event.psi)  * ((maxDistance  **  2) / (event.distance  ** 2))
    if z > 1:
        detectedEventList.append(event)

numDetectedEvents = len(detectedEventList) 

detectedThetaList = [event.theta for event in detectedEventList]
detectedPhiList = [event.phi for  event in detectedEventList]
detectedXList = [event.X for event in detectedEventList]
detectedYList = [event.Y for event  in detectedEventList]
detectedThetaPhiPoints = [event.thetaPhiPoint for event in detectedEventList]
detectedXYPoints = [event.XYPoint for event in detectedEventList]

print(detectedXYPoints)
fig, ax = plt.subplots()
colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood']

numTrials = 300
startingShape = [[math.cos(theta), math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, 100)]

for i in range(0 , numTrials):
    circle = [[i * math.cos(theta), i * math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, 100)]

    path = mppath.Path(circle)
    
    pointsInside = path.contains_points(detectedXYPoints)
    numPointsInside = list(pointsInside).count(True)
    detectionFraction = numPointsInside / numDetectedEvents

    if not (.89 < detectionFraction < .91):
        continue

    patch = mpatches.PathPatch(path, fill = False, color = (random.random(), random.random(), random.random()), lw=2)
    ax.add_patch(patch)


#Scatter plot showing where all theta and phis were detected

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.scatter(detectedXList, detectedYList, s = 1)

# # set up orthographic map projection with
# # perspective of satellite looking down at 50N, 100W.
# # use low resolution coastlines.
# globeMap = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
# # draw coastlines, country boundaries, fill continents.
# globeMap.drawcoastlines(linewidth=0.25)
# globeMap.drawcountries(linewidth=0.25)
# globeMap.fillcontinents(color='green',lake_color='aqua')
# # draw the edge of the map projection region (the projection limb)
# globeMap.drawmapboundary(fill_color='aqua')
# # draw lat/lon grid lines every 30 degrees.
# globeMap.drawmeridians(np.arange(0,360,30))
# globeMap.drawparallels(np.arange(-90,90,30))
# # make up some data on a regular lat/lon grid.
# nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
# lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
# lons = (delta*np.indices((nlats,nlons))[1,:,:])
# wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
# mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
# # compute native map projection coordinates of lat/lon grid.
# x, y = globeMap(lons*180./np.pi, lats*180./np.pi)
# # contour data over the map.
# cs = globeMap.contour(x,y,wave+mean,15,linewidths=1.5)
# plt.title('GW Detection fractions')
plt.show()

# delta = 100
# thetaList = np.linspace(-math.pi/2, math.pi/2, delta)
# phiList = np.linspace(-math.pi, math.pi, delta)

# psi = 0
# ZList, plotThetaList, plotPhiList, plotXList, plotYList = ([], [], [], [], [])
# for i, phi in enumerate(thetaList):
#     for j, theta in enumerate(phiList):
#         z = GWDetector.getSingleAntennaPowerPattern(theta, phi, psi)
#         plotThetaList.append(theta)
#         plotPhiList.append(phi)
#         plotXList.append(math.sin(theta) * math.cos(phi))
#         plotYList.append(math.sin(theta) * math.sin(phi))
#         ZList.append(z)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(plotXList, plotYList, ZList, c='b', marker='o')
# ax.plot_trisurf(plotXList, plotYList, ZList)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# fig, ax = plt.subplots()
# CS = ax.tricontour(plotXList, plotYList, ZList)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_xlabel(r"$\theta$")
# ax.set_ylabel(r"$\phi$")
# ax.set_title("Antenna Power Patterns for LIGO and VIRGO")
# plt.show()