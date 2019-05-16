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
from Circularization import *
from MiscFunctions import *

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "Main", "Images"))

# Event generation number of events generated is proportional to d^3 but we don't really care about the proportionality constant as it will be divided out eventually anyways


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


# Finding how many events at each distance
k = 1
maxDistance = 170
numDistancePoints = 200
distanceList = np.linspace(0, maxDistance, numDistancePoints)
numEventsAtDistance = []
for distance in distanceList:
    eventsForDistance = k * (distance ** 3)
    numEventsAtDistance.append(eventsForDistance)

approxTotalEvents = 1e4
tempTotalEvents = sum(numEventsAtDistance)
for i, distance in enumerate(distanceList):
    percentEvents = numEventsAtDistance[i] / tempTotalEvents
    numEventsAtDistance[i] = int(round(percentEvents * approxTotalEvents))

trueTotalEvents = sum(numEventsAtDistance)

# Event generation
eventList = []
for i, distance in enumerate(distanceList):
    for j in range(0, numEventsAtDistance[i]):
        theta, phi, psi = random.uniform(0, math.pi), random.uniform(
            0, 2 * math.pi), random.uniform(-math.pi, math.pi)
        eventList.append(Event(theta, phi, psi, distance))

detectedEventList = []
for i, event in enumerate(eventList):

    z = earthDetectorNetwork.getAntennaPowerPattern(
        event.theta, event.phi, event.psi) * ((maxDistance ** 2) / (event.distance ** 2))
    z *= ((1/8) * (1 + (6 * math.cos(event.psi) ** 2) + (math.cos(event.psi) ** 4)))
    if z > 12:
        detectedEventList.append(event)

    printProgressBar(i, len(eventList))

numDetectedEvents = len(detectedEventList)

detectedThetaList = [event.theta for event in detectedEventList]
detectedPhiList = [event.phi for event in detectedEventList]
detectedXList = [event.X for event in detectedEventList]
detectedYList = [event.Y for event in detectedEventList]
detectedThetaPhiPoints = [event.thetaPhiPoint for event in detectedEventList]
detectedXYPoints = [event.XYPoint for event in detectedEventList]

#Plotting inits
fig, ax = plt.subplots()

#Globemap
m = Basemap(projection='hammer',lon_0=0,resolution='c')
m.bluemarble(scale = .2)

#For testing purposes
#Transforming to to lattitude and longitude coordinates
detectedLonList = [math.degrees(phi) for phi in detectedPhiList]
detectedLatList = [math.degrees(theta - math.pi/2) for theta in detectedThetaList]
detectedLonList, detectedLatList = m(detectedLonList, detectedLatList)
detectedLonLatList = list(zip(detectedLonList, detectedLatList))
ax.scatter(detectedLonList, detectedLatList, s=.5, c='r')

#Fractional items 
_ret = getFractionalItems(detectedLonLatList, .5, ax, returnFmt=-2, refinements=5)
fracLonLatEdgePoints, fracLonLatEdgePath, fracLonLatPoints = _ret[0][0], _ret[0][1], _ret[1]

#Inside points
fracLonPointsList = [var[0] for var in fracLonLatPoints]
fracLatPointsList = [var[1] for var in fracLonLatPoints]
ax.scatter(fracLonPointsList, fracLatPointsList, s=.5, c='m')

#Edge points
fracLonEdgeList = [var[0] for var in fracLonLatEdgePoints]
fracLatEdgeList = [var[1] for var in fracLonLatEdgePoints]
fracLonLatEdgePoints = list(zip(fracLonEdgeList, fracLatEdgeList))

#Path Creation
codes = [mppath.Path.LINETO] * len(fracLonLatEdgePoints)
codes[0] = mppath.Path.MOVETO
codes[-1] = mppath.Path.CLOSEPOLY
fracLonLatEdgePath = mppath.Path(fracLonLatEdgePoints, codes)

#Creating Patches
fracLonLatEdgePatch = mpatches.PathPatch(fracLonLatEdgePath, fill=False, color='g', lw=2)
ax.add_patch(fracLonLatEdgePatch)

#Plotting Detectors
WashingtonLat, WashingtonLon = m(math.degrees(lambd_WASH), math.degrees(beta_WASH))
LouisianaLat, LouisianaLon = m(math.degrees(lambd_LOUIS), math.degrees(beta_LOUIS))
VirgoLat, VrigoLon = m(math.degrees(lambd_VIRGO), math.degrees(beta_VIRGO))
ax.scatter(WashingtonLat, WashingtonLon, c='y', s=4)
ax.scatter(LouisianaLat, LouisianaLon, c='y', s=4)
ax.scatter(VirgoLat, VrigoLon, c='y', s=4)

plt.show()



# #Plot detection contour
# _ret = getFractionalItems(detectedThetaPhiPoints, .5, returnFmt=0, refinements=5)
# fracThetaPhiEdge, fracThetaPhiEdgePath = _ret[0], _ret[1]
# fracLonsEdge = [math.degrees(coord[0] - math.pi/2) for coord in fracThetaPhiEdge]
# fracLatsEdge = [math.degrees(coord[1] - math.pi) for coord in fracThetaPhiEdge]
# x, y = m(fracLatsEdge, fracLonsEdge)
# fracLongsLatsEdge = list(zip(x,y))
# fracLongsLatsEdge = [list(coord) for coord in fracLongsLatsEdge]
# #making codes 
# codes = [mppath.Path.LINETO] * len(fracThetaPhiEdge)
# codes[0] = mppath.Path.MOVETO
# codes[-1] = mppath.Path.CLOSEPOLY
# #Creating path and patches
# fracPath = mppath.Path(fracLongsLatsEdge, codes)
# fracPatch = mpatches.PathPatch(fracPath, fill=False, color='g', lw=2)
# ax.add_patch(fracPatch)

# lats, lons = [math.degrees(theta - math.pi/2) for theta in detectedThetaList], [math.degrees(phi - math.pi) for phi in detectedPhiList]
# print(lons)
# x, y = m(lons, lats)  # transform coordinates
# ax.scatter(x, y, s = .1, c = 'r')
# plt.show()