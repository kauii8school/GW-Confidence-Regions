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
from scipy import interpolate 
from collections import OrderedDict

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

approxTotalEvents = 1e6
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
    if z > 12: #NOTE Change to 12 when using all detectors
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
m.bluemarble(scale = 1)

#For testing purposes
#Transforming to to lattitude and longitude coordinates
detectedLonList = [math.degrees(phi) for phi in detectedPhiList]
detectedLatList = [math.degrees(theta - math.pi/2) for theta in detectedThetaList]
detectedLonLatList = list(zip(detectedLonList, detectedLatList))

#Fractional items 
detectionFractionList = [.3, .5, .9]
sharpnessList = [2.3, 2.3, 2.3]
numFrac = len(detectionFractionList)
clusterDictDict = {}
for i, detectionFraction in enumerate(detectionFractionList):
    frac = Circularization(detectedLonLatList, detectionFraction)
    nClusters = 2
    clusterDict = frac.greedyAngleHeuristicMultiModal(nClusters, sharpnessList[i])
    clusterDictDict[detectionFraction] = clusterDict

#Edge points, also applies corner cutting
fracEdgePointsDict = {}
fracAreaDict = {}
for detectionFraction in detectionFractionList:
    edgePointsList = []
    clusterDict = clusterDictDict[detectionFraction]
    areaList = []
    for i in range(0, nClusters):
        hull =  ConvexHull(clusterDict[i])
        hullVertices = np.array([clusterDict[i][vertex] for vertex in hull.vertices])

        #Corner cutting
        #cutHullVertices = chaikins_corner_cutting(hullVertices, 12)

        #Interpolated curves
        tck, u = interpolate.splprep(hullVertices.T, u=None, s=500, per=1) 
        u_new = np.linspace(u.min(), u.max(), 10000)
        xNew, yNew = interpolate.splev(u_new, tck, der=0)

        #Calculating area
        areaList.append(0)

        interptHullVertices = list(zip(xNew, yNew))

        edgePointsList.append(interptHullVertices) #Cut hull vertieices is a list itself!
    
    #fracAreaDict[detectionFraction] = sum(areaList) / totalPossArea
    fracEdgePointsDict[detectionFraction] = edgePointsList

#Creating paths patches and plotting them
ecList = ['orange', 'green', 'blue']
for i, detectionFraction in enumerate(detectionFractionList):
    edgePointsList = fracEdgePointsDict[detectionFraction]
    for j, fracLonLatEdgePoints in enumerate(edgePointsList):
        x,y = zip(*fracLonLatEdgePoints)
        x,y = m(x,y)


        if detectionFraction >= .8 and j == 1:
            fracLonLatEdgePointsT = list(zip(x, y))
            fracLonLatEdgePointsT = [val for val in fracLonLatEdgePointsT if val[0] > 1e7]

            temp = [val for val in fracLonLatEdgePointsT if val[0] < 1e7]

            path = mppath.Path(temp) 
            patch = mpatches.PathPatch(path, lw=2, fill=False, ec = ecList[i], label = "detection fraction-{}".format(round(1 - detectionFraction, 1)))
            ax.add_patch(patch)

        else:
            fracLonLatEdgePointsT = list(zip(x,y))

        codes = [mppath.Path.LINETO] * len(fracLonLatEdgePointsT)
        codes[0] = mppath.Path.MOVETO
        codes[-1] = mppath.Path.CLOSEPOLY

        path = mppath.Path(fracLonLatEdgePointsT, codes) 
        #patch = mpatches.PathPatch(path, lw=2, fill=False, ec = ecList[i], label = "detection fraction-{}   fraction of earth-{}".format(round(1 - detectionFraction, 1), fracAreaDict[detectionFraction]))
        patch = mpatches.PathPatch(path, lw=2, fill=False, ec = ecList[i], label = "detection fraction-{}".format(round(1 - detectionFraction, 1)))
        ax.add_patch(patch)
        m.scatter(x, y, c='k', s=7)

#Plotting
#Plotting detections
detectedLonList, detectedLatList = m(detectedLonList, detectedLatList)
m.scatter(detectedLonList, detectedLatList, s = .5, c='r')


#Plotting Detectors
WashingtonLat, WashingtonLon = m(math.degrees(lambd_WASH), math.degrees(beta_WASH))
LouisianaLat, LouisianaLon = m(math.degrees(lambd_LOUIS), math.degrees(beta_LOUIS))
VirgoLat, VrigoLon = m(math.degrees(lambd_VIRGO), math.degrees(beta_VIRGO))
ax.scatter(WashingtonLat, WashingtonLon, c='y', s=7)
ax.scatter(LouisianaLat, LouisianaLon, c='y', s=7)
ax.scatter(VirgoLat, VrigoLon, c='y', s=7)


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc = 2)
plt.show()