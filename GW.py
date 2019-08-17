import math
import numpy as np


def testFunction(theta, phi):
    return np.sin(theta) * np.sin(phi)

class GWDetectorNetwork:

    def __init__(self, detectorList):

        """ Takes a list of dictionaries which contain data for detectors and finds the antenna power patterns and detection values for them """

        #Converts detector settings list into detector objects
        self.detectorList = []
        for detector in detectorList:
            detectorObject = GWDetector(detector["beta"], detector["lambd"], detector["chi"], detector["eta"], detector["name"], detector["visibility distance"])
            self.detectorList.append(detectorObject)

    def getAntennaPowerPattern(self, theta, phi, psi):

        """ sums all antenna pattern """
        
        antennaPowerPatternList = []
        for detector in self.detectorList:
            antennaPowerPatternList.append(detector.getAntennaPowerPattern(theta, phi, psi))
        
        return sum(antennaPowerPatternList)
       

class GWDetector:

    """ Class for a GW detector contains antenna power patterns mostly, params include longitude, lattitude, orientation, and seperation """        

    def __init__(self, _beta, _lambda, _chi, _eta, _name, _horizonDistance):

        """ Beta is lattitude, lambda is longitude, chi is orientation measured counter clockwise from east, opening angle of arms is eta """

        self.beta = _beta
        self.lambd = _lambda
        self.chi = _chi
        self.eta = _eta
        self.name = _name
        self.dv = _horizonDistance

        self.appList, self.crossList, self.plusList, self.aList, self.bList = ([], [], [], [], [])


    def getAntennaPowerPattern(self, theta, phi, psi):
        a = self.afunction(theta, phi)
        b = self.bfunction(theta, phi)

        plus = self.AP_PLUS(a, b, psi)
        cross = self.AP_CROSS(a, b, psi)

        powerPattern = (plus ** 2) + (cross ** 2)
        self.appList.append(powerPattern)
        return powerPattern

    def afunction(self, theta, phi):

        ret = ((1/16) * math.sin(2 * self.chi) * (3 - math.cos(2 * self.beta)) * (3 - math.cos(2 * theta))\
        * math.cos(2 * (phi + self.lambd)))\
        + ((1/4) * math.cos(2 * self.chi) * math.sin(self.beta) * (3 - math.cos(2 * theta))\
        * math.sin(2 * (phi + self.lambd)))\
        + ((1/4) * math.sin(2 * self.chi) * math.sin(2 * self.beta)\
        * math.sin(2 * theta) * math.cos(phi + self.lambd))\
        + ((1/2) * math.cos(2 * self.chi) * math.cos(self.beta)\
        * math.sin(2 * theta) * math.sin(phi + self.lambd))\
        + ((3/4) * math.sin(2 * self.chi) * (math.cos(self.beta) ** 2) * (math.sin(theta) ** 2))

        self.aList.append(ret)

        return ret

    def bfunction(self, theta, phi):

        ret = (math.cos(2 * self.chi) * math.sin(self.beta)\
        * math.cos(theta) * math.cos(2 * (phi + self.lambd)))\
        - ((1/4) * math.sin(2 * self.chi) * (3 - math.cos(2 * self.beta))\
        * math.cos(theta) * math.sin(2 * (phi + self.lambd)))\
        + (math.cos(2 * self.chi) * math.cos(self.beta)\
        * math.sin(theta) * math.cos(phi + self.lambd))\
        - ((1/2) * math.sin(2 * self.chi) * math.sin(2 * self.beta)\
        * math.sin(theta) * math.sin(phi + self.lambd))\

        self.bList.append(ret)

        return ret

    def AP_PLUS(self, a, b, psi):
        ret = math.sin(self.eta) * ((a * math.cos(2 * psi)) + (b * math.sin(2 * psi)))

        self.plusList.append(ret)

        return ret


    def AP_CROSS(self, a, b, psi):
        ret = math.sin(self.eta) * ((b * math.cos(2 * psi)) - (a * math.sin(2 * psi)))

        self.crossList.append(ret)

        return ret

    @staticmethod
    def Single_AP_PLUS(theta, phi, psi):
        ret = ((1/2) * (1 + (math.cos(theta) ** 2)) * math.cos(2 * phi) * math.cos(2 * psi)) - (math.cos(theta) * math.sin(2 * phi) * math.sin(2 *\
            psi))
        return ret

    @staticmethod
    def Single_AP_CROSS(theta, phi, psi):
        ret = ((1/2) * (1 + (math.cos(theta) ** 2)) * math.cos(2 * phi) * math.sin(2 * psi)) + (math.cos(theta) * math.sin(2 * phi) * math.cos(2 *\
        psi))
        return ret

    @staticmethod
    def getSingleAntennaPowerPattern(theta, phi, psi):
        ret = (GWDetector.Single_AP_CROSS(theta, phi, psi) ** 2) + (GWDetector.Single_AP_PLUS(theta, phi, psi) ** 2)
        return ret 

    @staticmethod
    def inclinationMultiplier(psi):
        return (1/8) * (1 + (6 * (math.cos(psi) ** 2)) + (math.cos(psi) ** 4))

    @staticmethod
    def DMS_TO_DEGREES(degs, mins, secs):
        ret = degs + (mins/60) + (secs/3600)
        return ret

#GW Detectors
#chi is orientation from East in degrees
#input different detector locations and orientations, beta and lambda are lattitude and longitude
eta_AP = math.pi/2

#Livingston Louisiana LIGO
beta_LOUIS = math.radians(GWDetector.DMS_TO_DEGREES(30,33,46.4))
lambd_LOUIS = math.radians(GWDetector.DMS_TO_DEGREES(-90,-46,-27.3))
chi_LOUIS = math.radians(208)
Dv_LOUIS = 190 
LouisianaDict = {"beta" : beta_LOUIS, "lambd" : lambd_LOUIS, "chi" : chi_LOUIS, "eta" : eta_AP, "name" : "LIGO Louisiana", "visibility distance" : Dv_LOUIS}

#Hanford Washington LIGO
beta_WASH = math.radians(GWDetector.DMS_TO_DEGREES(46,27,18.5))
lambd_WASH = math.radians(GWDetector.DMS_TO_DEGREES(-119,-24,-27.6))
chi_WASH = math.radians(279)
Dv_WASH = 190
WashingtonDict = {"beta" : beta_WASH, "lambd" : lambd_WASH, "chi" : chi_WASH, "eta" : eta_AP, "name" : "LIGO Washington", "visibility distance" : Dv_WASH}

#VIRGO
beta_VIRGO = math.radians(GWDetector.DMS_TO_DEGREES(43,37,53))
lambd_VIRGO = math.radians(GWDetector.DMS_TO_DEGREES(10,30,16))
chi_VIRGO = math.radians(333.5)
Dv_VIRGO = 170
VirgoDict = {"beta" : beta_VIRGO, "lambd" : lambd_VIRGO, "chi" : chi_VIRGO, "eta" : eta_AP, "name" : "VIRGO Italy", "visibility distance" : Dv_VIRGO}

#KAGRA
beta_KAGRA = math.radians(GWDetector.DMS_TO_DEGREES(36, 15, 0))
lambd_KAGRA = math.radians(GWDetector.DMS_TO_DEGREES(137, 10, 48))
chi_KAGRA = math.radians(20.)
Dv_KAGRA = 170
KAGRADict = {"beta" : beta_KAGRA, "lambd" : lambd_KAGRA, "chi" : chi_KAGRA, "eta" : eta_AP, "name" : "KAGRA Japan", "visibility distance" : Dv_KAGRA}

#Test detector
beta_Test = math.radians(GWDetector.DMS_TO_DEGREES(0, 0, 0))  + math.pi/2
lambd_Test = math.radians(GWDetector.DMS_TO_DEGREES(0, 0, 0))
chi_Test = 0
Dv_Test = 170
TestDict = {"beta" : beta_Test, "lambd" : lambd_Test, "chi" : chi_Test, "eta" : eta_AP, "name" : "VIRGO Italy", "visibility distance" : Dv_Test}

Virgo = GWDetector(VirgoDict["beta"], VirgoDict["lambd"], VirgoDict["chi"], VirgoDict["eta"], VirgoDict["name"], VirgoDict["visibility distance"])

Test = GWDetector(TestDict["beta"], TestDict["lambd"], TestDict["chi"], TestDict["eta"], TestDict["name"], TestDict["visibility distance"])

Washington = GWDetector(WashingtonDict["beta"], WashingtonDict["lambd"], WashingtonDict["chi"], WashingtonDict["eta"], WashingtonDict["name"], WashingtonDict["visibility distance"])

Louisiana = GWDetector(LouisianaDict["beta"], LouisianaDict["lambd"], LouisianaDict["chi"], LouisianaDict["eta"], LouisianaDict["name"], LouisianaDict["visibility distance"])

detectorToTest = Virgo

detectorDictList = [VirgoDict, WashingtonDict, LouisianaDict]
earthDetectorNetwork = GWDetectorNetwork(detectorDictList)