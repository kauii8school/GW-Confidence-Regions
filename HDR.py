import math
import numpy as np
import matplotlib.path as mppath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.path as mppath
from scipy.spatial import ConvexHull
from MiscFunctions import *

def getFractionalItems(points, detectionFraction, ax, returnFmt = 0, refinements = 1):

    """ 
    Function that hones in on closest density of points and then finds all near points to it by calculating the average of those points and then 
    appending the closest to the average to the list. It will return one of the following things:
        -1). Fractional detection points
         0). [list(tuples), mppath.Path] of cut fractional points
         1). [list(tuples), mppath.Path] of uncut (Polygon) points
         2). [[list(tuples), mppath.Path], [list(tuples), mppath.Path]] fractional cut and uncut respectively
    """
    #Inputs: detection fraction and set of points
    #Outputs: mppath.Path giving line circling fractional amount of points

    def fractionalPoints(totalNodeList, recNodeList, fracPoints):

        """ Starts out with one point should be in a place of high density #NOTE this is not automated yet. Keep adding points (it will add the closest)
        point to the set over and over until 50% of the points are encircled. Then it will return a list of those points """

        avPoint = averagePoints(recNodeList)

        for i in range(0, fracPoints):
            closestPoint = closest_node(avPoint, totalNodeList) #Finds closest point
            totalNodeList.remove(closestPoint)
            recNodeList.append(closestPoint)

            printProgressBar(i, fracPoints)    

        return recNodeList 

    #Gets fractional points 
    numPointsFrac = math.floor(len(points) * detectionFraction)
    fracPoints = fractionalPoints(points, [(math.radians(30), math.radians(-119))], numPointsFrac)
    ax.scatter(math.radians(30), math.radians(-119), c='orangered')
    
    #Hull creation and getting of verticies
    hull = ConvexHull(fracPoints)
    polyVertices = [fracPoints[vertex] for vertex in hull.vertices] 
    hullArea = hull.area

    #Path creation 
    polyCodes = [mppath.Path.LINETO] * len(polyVertices)
    polyCodes[0] = mppath.Path.MOVETO
    polyCodes[-1] = mppath.Path.CLOSEPOLY

    cutCodes = [mppath.Path.LINETO] * len(cutVertices)
    cutCodes[0] = mppath.Path.MOVETO
    cutCodes[-1] = mppath.Path.CLOSEPOLY

    polyPath = mppath.Path(polyVertices, polyCodes)
    cutPath = mppath.Path(cutVertices, cutCodes)

    #How you want the information returned 
    if returnFmt == -2:
        return [[cutVertices, cutPath], fracPoints]
    if returnFmt == -1:
        return fracPoints
    if returnFmt == 0:
        return [cutVertices, cutPath]
    if returnFmt == 1:
        return [polyVertices, polyPath]
    if returnFmt == 2:
        return [[cutVertices, cutPath], [polyVertices, polyPath]]

# points = points()
# numPoints = len(points)
# detectionFraction = .5
# numPointsFrac = math.floor(numPoints * detectionFraction)
# fracPoints = getFractionalItems(points, numPointsFrac, -1)
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# xPoints = [point[0] for point in points]
# yPoints = [point[1] for point in points]
# plt.scatter(xPoints, yPoints, s=1)
# fracXPoints = [point[0] for point in fracPoints]
# fracYPoints = [point[1] for point in fracPoints]
# plt.scatter(fracXPoints, fracYPoints, s = 1)
# plt.show()