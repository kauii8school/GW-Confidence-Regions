import numpy as np
import math
import matplotlib.path as mppath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import poly_point_isect
import random
import bisect

class Blob:

    @staticmethod
    def PolyArea(polyPoints):

        """ Given set of points of polynomial calculates area using shoelace formula """

        x, y = zip(*polyPoints)
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def __init__(self, points, numPoints, delta):
        self.state = [[.2 * math.cos(theta), .2 * math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, numPoints)]

        self.points = points
        self.xList = list(np.linspace(-1, 1, delta))
        self.yList = list(np.linspace(-1, 1, delta))
        
        #Adds circle values into list so we can move to them
        for point in self.state:
            x = point[0]
            y = point[1]
            bisect.insort(self.xList, x)
            bisect.insort(self.yList, y)

        self.numPoints = numPoints

        #Generate code so it moves and also closes properly
        self.codes = [mppath.Path.LINETO] * len(self.state)
        self.codes[0] = mppath.Path.MOVETO
        self.codes[-1] = mppath.Path.CLOSEPOLY

        self.reward = 0
        self.count = 0
        self.render()

    def move(self, action):

        """ Move the circle chosen circle dot and return a new state """

        pointToMove = action[0]
        direction = action[1]
        possState = []

        oldPoint = self.state[pointToMove]

        
        
        if direction == "u":
            nextY = self.yList[self.yList.index(oldPoint[1]) + 1]
            x = oldPoint[0]
            y = nextY
            
        elif direction == "d":
            prevY = self.yList[self.yList.index(oldPoint[1]) - 1]
            x = oldPoint[0]
            y = prevY

        elif direction == "r":
            nextX = self.xList[self.xList.index(oldPoint[0]) + 1]
            x = nextX
            y = oldPoint[1]

        elif direction == "l":
            prevX = self.xList[self.xList.index(oldPoint[0]) - 1]
            x = prevX
            y = oldPoint[1]

        newPoint = [x, y]

        for i, point in enumerate(self.state):
            if i == pointToMove:
                possState.append(newPoint)
            else:
                possState.append(point)

        return possState


    def step(self, action):
        
        """ Performs step and returns new state and reward. Action[0] = point to move Action[1] = point to move it to """

        #Moves the point to the new point, but generates a possible state first to check if it contains polygon intersections

        possState =  self.move(action)
        # print(possState)
        
        isect = poly_point_isect.isect_polygon(tuple(possState))
        #NOTE closing the polynomial sometimes causes intersection that the Bentely-Ottoman Sweep-line algo does not detect, fix this later
        
        if not len(isect) == 0:
            #Invalid move since it caused intersection
            return [self.state, "INVALID"]
        else:
            self.state = possState
        
        #Calculates detection fraction encloses
        path = mppath.Path(self.state, self.codes)
        pointsInside = path.contains_points(self.points)
        numPointsInside = list(pointsInside).count(True)
        detectionFraction = numPointsInside / len(self.points)
        difference = .5 - detectionFraction

        #Calculates area enclosed
        area = self.PolyArea(self.state)

        self.reward = ((1 - difference) ** 2) + ((area / 4) * .25)


        self.count += 1 
        return [self.state, self.reward]

    def reset(self, points):
        
        """ Resets back to a cirlce """
        self.state = [[.2 * math.cos(theta), .2 * math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, self.numPoints)]
        self.points = points
        self.render()

    def render(self, mode='human', close=False, label = None):

        """ saves matplotlib fig to disk """
        
        fig, ax = plt.subplots()
        path = mppath.Path(self.state, self.codes)
        plt.scatter([point[0] for point in self.points], [point[1] for point in self.points], s = 1)
        patch = mpatches.PathPatch(path, fill = False, color = 'r', lw=2, label = label)
        ax.add_patch(patch)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig("/home/n/Documents/Research/GW-Contour-Mapping/Frames/Frame_{0:03}".format(self.count))
        plt.close()