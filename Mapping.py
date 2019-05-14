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
        theta, phi, psi = random.uniform(0, math.pi), random.uniform(0, 2 * math.pi), random.uniform(-math.pi, math.pi)
        eventList.append(Event(theta, phi, psi, distance))

detectedEventList = []
for i, event in enumerate(eventList):

        z = GWDetector.getSingleAntennaPowerPattern(event.theta, event.phi, event.psi)  * ((maxDistance  **  2) / (event.distance  ** 2))
        z *= ((1/8) * (1 + (6 * math.cos(event.psi) ** 2) + (math.cos(event.psi) ** 4)))
        if z > 1:
                detectedEventList.append(event)

        printProgressBar(i, len(eventList))

numDetectedEvents = len(detectedEventList) 

detectedThetaList = [event.theta for event in detectedEventList]
detectedPhiList = [event.phi for  event in detectedEventList]
detectedXList = [event.X for event in detectedEventList]
detectedYList = [event.Y for event  in detectedEventList]
detectedThetaPhiPoints = [event.thetaPhiPoint for event in detectedEventList]
detectedXYPoints = [event.XYPoint for event in detectedEventList]

fig, ax = plt.subplots()
colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood']

numTrials = 300
startingShape = [[math.cos(theta), math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, 100)]

_ret = getFractionalItems(detectedXYPoints, .5, returnFmt=1)
path = _ret[1] 

patch = mpatches.PathPatch(path, fill = False, color = (random.random(), random.random(), random.random()), lw=2)
ax.add_patch(patch)


#Scatter plot showing where all theta and phis were detected

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

ax.scatter(detectedXList, detectedYList, s = 1)
plt.show()
# fig = plt.gcf()
# fig.set_size_inches(8, 6.5)

# m = Basemap(projection='merc', \
#             llcrnrlat=-80, urcrnrlat=80, \
#             llcrnrlon=-180, urcrnrlon=180, \
#             lat_ts=20, \
#             resolution='c')

# # m = Basemap(projection='ortho',lat_0=0,lon_0=0,resolution='l') #Globe Projection

# # m.bluemarble(scale=0.2)   # full scale will be overkill
# m.drawcoastlines(color='black', linewidth=0.2)  # add coastlines

# lons, lats = [math.degrees(theta - math.pi/2) for theta in detectedThetaList], [math.degrees(phi) for phi in detectedPhiList]
# print(math.degrees(-math.pi/2))
# x, y = m(lons, lats)  # transform coordinates
# plt.scatter(x, y, s=1) 

# plt.show()