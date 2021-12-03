import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
from skimage.measure import label, regionprops
from sklearn.neighbors import NearestNeighbors


def findMaxVolumeAndPoints(mask_, numPoints_):
    # The largest area
    props = regionprops(label(np.float64(np.asanyarray(mask_))))
    areaList = []
    for prop in props:
        areaList.append(prop.area)
        print(prop.area)

    maxIndex = np.argmax(areaList)
    targetProp = props[maxIndex]
    #     # debug
    #     plt.figure()
    #     plt.imshow(targetProp.image)

    # New image generation:
    refinedImg = np.zeros(np.shape(mask_))
    refinedImg[targetProp.bbox[0]:targetProp.bbox[2], targetProp.bbox[1]:targetProp.bbox[3]] = targetProp.convex_image # "convex_image" is recommend for approx.
    refinedImg = np.uint8(refinedImg)
    # edge image and their coordinates
    edges = cv2.Canny(refinedImg, 0, 1)
    plt.figure()
    plt.imshow(edges)
    indices = np.where(edges != [0])
    coordinates = np.column_stack((indices[0], indices[1]))

    # point order optm.
    x = indices[0]
    y = indices[1]
    points = np.c_[x, y]

    # clf = NearestNeighbors(2).fit(points)
    neigh = NearestNeighbors(n_neighbors=3, radius=0.5)
    clf = neigh.fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    mindist = np.inf
    minidx = 0

    for i in range(len(points)):
        p = paths[i]  # order of nodes
        ordered = points[p]  # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i

    opt_order = paths[minidx]
    xx = x[opt_order]
    yy = y[opt_order]


    stepSizeTemp = np.uint8(np.round(len(xx) / numPoints_))

    xxExtracted = xx[::stepSizeTemp]
    yyExtracted = yy[::stepSizeTemp]

    if len(xxExtracted) == numPoints_+1:
        xxExtracted = list(xxExtracted)
        yyExtracted = list(yyExtracted)
        xxExtracted.pop()
        yyExtracted.pop()
        xxExtracted = tuple(xxExtracted)
        yyExtracted = tuple(yyExtracted)

    coordinatesF = np.column_stack((xxExtracted, yyExtracted))
    volumeF = targetProp.area

    # visual debug
    # plt.figure()
    # plt.plot(coordinatesF[:, 0], coordinatesF[:, 1])
    # plt.plot(xx, yy)
    # plt.show()

    return volumeF, coordinatesF

##
jsonPath = r"C:\Users\admin\Downloads\Graspers (straight) (OLYMPUS) (GRSL-UCOL)\SNUH_DC07_JCW0_RALP_0031_00098526.json"


f_jason = open(jsonPath, 'rt', encoding='UTF8')
jsonData =json.load(f_jason)
jsonPoints= jsonData.get('annotations')[0].get('points')
jsonImgSize = jsonData.get('images')


points = np.array(jsonPoints, dtype=np.int)
mask = np.zeros((jsonImgSize.get('height'), jsonImgSize.get('width')), dtype = np.uint8)
cv2.fillPoly(mask,[points], 1)

volTemp, coordTemp = findMaxVolumeAndPoints(mask, 10)

plt.figure()
plt.imshow (mask)
plt.plot(coordTemp[:,1], coordTemp[:,0])
plt.show()

