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
from pathlib import Path

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

approxTotalEvents = 5e4
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
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.bluemarble(scale = 1)

#For testing purposes
#Transforming to to lattitude and longitude coordinates
detectedLonList = [math.degrees(phi) for phi in detectedPhiList]
detectedLatList = [math.degrees(theta - math.pi/2) for theta in detectedThetaList]
detectedLonLatList = list(zip(detectedLonList, detectedLatList))

#Fractional items 
detectionFractionList = [.33, .5, .66]
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
totalPossArea = 510.1e12
for detectionFraction in detectionFractionList:
    edgePointsList = []
    clusterDict = clusterDictDict[detectionFraction]
    areaList = []
    for i in range(0, nClusters):
        hull =  ConvexHull(clusterDict[i])
        hullVertices = np.array([clusterDict[i][vertex] for vertex in hull.vertices])

        #Interpolated curves
        tck, u = interpolate.splprep(hullVertices.T, u=None, s=300, per=1) 
        u_new = np.linspace(u.min(), u.max(), 10000)
        xNew, yNew = interpolate.splev(u_new, tck, der=0)

        #Calculating area
        areaList.append(projectionArea(hullVertices))

        interptHullVertices = list(zip(xNew, yNew))

        edgePointsList.append(interptHullVertices) #Cut hull vertieices is a list itself!
    
    fracAreaDict[detectionFraction] = sum(areaList) / totalPossArea
    fracEdgePointsDict[detectionFraction] = edgePointsList

#Creating paths patches and plotting them
ecList = ['darkred', 'red', 'orange']
for i, detectionFraction in enumerate(detectionFractionList):
    edgePointsList = fracEdgePointsDict[detectionFraction]
    for j, fracLonLatEdgePoints in enumerate(edgePointsList):
        lon, lat = zip(*fracLonLatEdgePoints)

        x,y = m(lon,lat)

        #Not generalized but makes it so path doesn't go through middle of projection and instead around
        if detectionFraction >= .7 and max(lon) < 300:
            fracLonLatEdgePointsTemp = list(zip(x, y))
            fracLonLatEdgePointsT = [val for val in fracLonLatEdgePointsTemp if val[0] < 1.5e7]

            temp = [val for val in fracLonLatEdgePointsTemp if val[0] > 1.5e7]

            x, y = zip(*temp)
            path = mppath.Path(temp, closed=False, readonly = False) 
            patch = mpatches.PathPatch(path, lw=2, fill=False, ec = ecList[i], label = "detection fraction-{}   area-{}".format(detectionFraction, round(fracAreaDict[detectionFraction], 3)))
            ax.add_patch(patch)

        else:
            fracLonLatEdgePointsT = list(zip(x,y))

        codes = [mppath.Path.LINETO] * len(fracLonLatEdgePointsT)
        codes[0] = mppath.Path.MOVETO
        codes[-1] = mppath.Path.STOP


        path = mppath.Path(fracLonLatEdgePointsT, codes, closed=False, readonly = False) 
        patch = mpatches.PathPatch(path, lw=2, fill=False, ec = ecList[i], label = "detection fraction: {}%   area: {}%".format(int(detectionFraction * 100), int(round(fracAreaDict[detectionFraction] * 100, 0))))
        ax.add_patch(patch)


# #Plotting
# #Plotting detections
# detectedLonList, detectedLatList = m(detectedLonList, detectedLatList)
# m.scatter(detectedLonList, detectedLatList, s = .5, c='r')


# #Plotting Detectors
zorderDetectors = 10
detectorColor = 'yellow'
fontSize = 15
WashingtonLat, WashingtonLon = m(math.degrees(lambd_WASH), math.degrees(beta_WASH))
LouisianaLat, LouisianaLon = m(math.degrees(lambd_LOUIS), math.degrees(beta_LOUIS))
VirgoLat, VirgoLon = m(math.degrees(lambd_VIRGO), math.degrees(beta_VIRGO))

ax.annotate("LIGO Hanford", (WashingtonLat + 2e5, WashingtonLon + 2e5), color=detectorColor, zorder=zorderDetectors, size=fontSize)
ax.scatter(WashingtonLat, WashingtonLon, c=detectorColor, s=11, zorder=zorderDetectors)
ax.annotate("LIGO Livingston", (LouisianaLat + 2e5, LouisianaLon + 2e5), color=detectorColor, zorder=zorderDetectors, size=fontSize)
ax.scatter(LouisianaLat, LouisianaLon, c=detectorColor, s=11, zorder=zorderDetectors)
ax.annotate("VIRGO", (VirgoLat + 2e5, VirgoLon + 2e5), color=detectorColor, zorder=zorderDetectors, size=fontSize)
ax.scatter(VirgoLat, VirgoLon, c=detectorColor, s=11, zorder=zorderDetectors)

beta_Cheren_S = math.radians(GWDetector.DMS_TO_DEGREES(-24,-41,-.34))
lambd_Cheren_S = math.radians(GWDetector.DMS_TO_DEGREES(-70,-18,-58.84))
CherenSLat, CherenSLon = m(math.degrees(lambd_Cheren_S), math.degrees(beta_Cheren_S))
ax.annotate("CTA North", (CherenSLat + 2e5, CherenSLon + 2e5), color=detectorColor, zorder=zorderDetectors, size=fontSize)
ax.scatter(CherenSLat, CherenSLon, c=detectorColor, s=25, marker='*', zorder=zorderDetectors)

beta_Cheren_N = math.radians(GWDetector.DMS_TO_DEGREES(28,45,43.7904))
lambd_Cheren_N = math.radians(GWDetector.DMS_TO_DEGREES(-17,-53,-31.218))
CherenSLat, CherenSLon = m(math.degrees(lambd_Cheren_N), math.degrees(beta_Cheren_N))
ax.annotate("CTA South", (CherenSLat + 2e5, CherenSLon + 2e5), color=detectorColor, zorder=zorderDetectors, size=fontSize)
ax.scatter(CherenSLat, CherenSLon, c=detectorColor, s=25, marker='*', zorder=zorderDetectors)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.45, 0.2), fontsize = 13)
plt.show()


fn = Path('/home/n/Documents/Research/GW-Contour-Mapping/Map.svg').expanduser()
plt.draw() # necessary to render figure before saving
fig.savefig(fn, bbox_inches='tight')

