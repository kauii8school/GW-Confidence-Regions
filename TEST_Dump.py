import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# make up some data for scatter plot
lats = np.random.randint(-75, 75, size=20)
lons = np.random.randint(-179, 179, size=20)

fig = plt.gcf()
fig.set_size_inches(8, 6.5)

m = Basemap(projection='merc', \
            llcrnrlat=-80, urcrnrlat=80, \
            llcrnrlon=-180, urcrnrlon=180, \
            lat_ts=20, \
            resolution='c')

#m.bluemarble(scale=0.2)   # full scale will be overkill idk why not working fix later
# m.shadedrelief()
m.drawcoastlines(color='black', linewidth=0.2)  # add coastlines

x, y = m(lons, lats)  # transform coordinates
plt.scatter(x, y, 10, marker='o', color='Red') 

plt.show()

def closest_node(node, nodes):

    """ returns closest node """

    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# # setup Lambert Conformal basemap.
# # set resolution=None to skip processing of boundary datasets.
# m = Basemap(width=12000000,height=9000000,projection='lcc',
#             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# m.bluemarble()
# plt.show()

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
# plt.show()