import sys
import math
from random import randint
import cv2
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

SHOW_PALETTE     = True
SHOW_CLUSTER_PLT = True

k = 8
img = cv2.imread("octopus.jpg", cv2.IMREAD_COLOR)

if SHOW_PALETTE:
    imgout = cv2.copyMakeBorder(img, 0, 0, 0, 20, cv2.BORDER_CONSTANT,value=[255,255,255])
else:
    imgout = img.copy()

h = img.shape[0]
w = img.shape[1]

if SHOW_CLUSTER_PLT:
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

# cluster means at the start of each iteration
means = [[0 for i in range(0, 3)] for j in range(0, k)]

# index of the nearest mean for each pixel
nearest_mean = [[0 for y in range(0, h)] for x in range(0, w)]
# centroid of new cluster
centroid = [[0 for i in range(0, 3)] for j in range(0, k)]
# number of elements in each cluster
members = [0] * k

# initialize means to random values
for i in range(0, k):
    means[i] = [randint(0, 255), # Blue
                randint(0, 255), # Green
                randint(0, 255)] # Red

# loop until convergence
iteration = 0
delta = 500
while delta > 5:
    # calculate pixel clusters based on current means
    print("[%02d] calculating pixel clusters       " % iteration, end="\r", flush=True)
    for x in range(0, w):
        for y in range(0, h):
            mdist = sys.maxsize
            # find closest mean
            for m in range(0, len(means)):
                d_b = img.item(y, x, 0) - means[m][0]
                d_g = img.item(y, x, 1) - means[m][1]
                d_r = img.item(y, x, 2) - means[m][2]
                d = (d_b * d_b) + (d_g * d_g) + (d_r * d_r)
                if d < mdist:
                    mdist = d
                    nearest_mean[x][y] = m

    # calculate the centroid of each cluster
    print("[%02d] finding centroids of new clusters" % iteration, end="\r", flush=True)
    for x in range(0, w):
        for y in range(0, h):
            i = nearest_mean[x][y]
            members[i] += 1
            for c in range(0, 3):
                centroid[i][c] += img.item(y, x, c)

    for i in range(0, k):
        if members[i] == 0:
            continue
        for c in range(0, 3):
            centroid[i][c] = int(centroid[i][c] / members[i])

    # test distance between new centroids and means
    print("[%02d] calculating delta                " % iteration, end="\r", flush=True)
    delta = 0
    for i in range(0, k):
        delta += ( pow(centroid[i][0] - means[i][0], 2)
                 + pow(centroid[i][1] - means[i][1], 2)
                 + pow(centroid[i][2] - means[i][2], 2) )
    delta = delta / k

    means = centroid[:]
    centroid = [[0 for i in range(0, 3)] for j in range(0, k)]
    members = [0] * k

    # generate output image
    print("[%02d] generating output image          " % iteration, end="\r", flush=True)
    for x in range(0, w):
        for y in range(0, h):
            i = nearest_mean[x][y]
            imgout[y][x] = means[i]

    # show color palette next to image
    if SHOW_PALETTE:
        interval = math.floor(h/k)
        for i in range(0, k):
            imgout[1+(i*interval):(i+1)*interval, w+1:-1] = means[i]

    # show cluster plot
    if SHOW_CLUSTER_PLT:
        ax.scatter(list(c[0] for c in means),
                   list(c[1] for c in means),
                   list(c[2] for c in means))
        pyplot.savefig("out/fig%02d.png" % iteration)

    cv2.imwrite("out/%02d.png" % iteration, imgout)
    iteration += 1

print("\nFinished in %d iterations." % iteration)
