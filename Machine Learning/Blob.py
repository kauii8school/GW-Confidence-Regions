import numpy as np
import math
import matplotlib.path as mppath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

class Blob:

    @staticmethod
    def PolyArea(polyPoints):

        """ Given set of points of polynomial calculates area using shoelace formula """

        x, y = zip(*polyPoints)
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def __init__(self, points, numPoints):
        self.state = [[.2 * math.cos(theta), .2 * math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, numPoints)]

        self.points = points
        self.reward = 0
        self.count = 0

    def step(self, action):
        
        """ Performs step and returns new state and reward. Action[0] = point to move Action[1] = point to move it to """

        #Moves the point to the new point
        pointToMove = action[0]
        newPoint = action[1]
        self.state[pointToMove] = newPoint
        
        #Calculates detection fraction enclosed
        path = mppath.Path(self.state)
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
        self.state = [[.2 * math.cos(theta), .2 * math.sin(theta)] for theta in np.linspace(0, 2 * math.pi, 15)]
        self.points = points

    def render(self, mode='human', close=False):

        """ saves matplotlib fig to disk """
        
        fig, ax = plt.subplots()
        path = mppath.Path(self.state)
        patch = mpatches.PathPatch(path, fill = False, color = 'r', lw=2)
        ax.add_patch(patch)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig("/home/n/Documents/Research/GW-Contour-Mapping/Frames/Frame_{0:03}".format(self.count))
        plt.close()