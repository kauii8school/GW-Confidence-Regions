from Circularization import *
#from TEST_Dump import points
cov = [[1, 0], [0, 1]]

points1 = np.random.multivariate_normal([-3, 3], cov, 300)
points2 = np.random.multivariate_normal([3, -3], cov, 300)
points3 = np.random.multivariate_normal([0, 0], cov, 600)
points1 = [list(element) for element in points1]
points2 = [list(element) for element in points2]
points3 = [list(element) for element in points3]

bimodalDist = points1 + points2
points = bimodalDist
# xPoints, yPoints = zip(*points)
# plt.scatter(xPoints, yPoints)
# plt.show()

testCirculizer = Circularization(points, .5)
greedHeur = testCirculizer.greedyAngleHeuristicMultiModal(2, math.pi/2)
hull = ConvexHull(points)
hullVertices = [points[vertex] for vertex in hull.vertices] 
x, y = zip(*hullVertices)
plt.scatter(x,y,c='r')
xPoints = [point[0] for point in points]
yPoints = [point[1] for point in points]
plt.scatter(xPoints, yPoints, s=1)

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()    