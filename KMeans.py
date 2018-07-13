from numpy import *
import time
import matplotlib.pyplot as plt
import sys


# calculate Euclidean distance  
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample  
        for i in range(numSamples):  # range
            minDist = 100000000.0
            minIndex = 0
            ## for each centroid  
            ## find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ##update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids  
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    ## Calculate Distance in Cluster
    i = 0
    size = zeros((k, 1))
    for i in range(k):
        pointsCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0]]
        size[i, 0] = len(pointsCluster)

    sum = 0
    i = 0
    j = 0
    DisInCluster = 0
    for j in range(k):
        sizeofdata = int(size[j, 0])
        for i in range(sizeofdata):
            points = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]][i]
            sum += euclDistance(points, centroids[j, :])/sizeofdata
    DisInCluster = sum/k

    print('cluster complete')

    ## Calculate Distance Between Cluster
    Dis = 0
    DisBtnCluster = 0
    i = 0
    j = 0
    for i in range(k):
        for j in range(j + 1, k):
            Dis += euclDistance(centroids[i, :], centroids[j, :])

    DisBtnCluster = 2 * Dis / (k * (k - 1))

    ## Calculate different num in each cluster
    i = 0
    j = 0
    diff = 0
    for i in range(k):
        for j in range(i+1,k):
            diff += abs(size[i, 0] - size[j, 0])

    fitness = 10*DisBtnCluster/(1+DisBtnCluster) + 45.5/208

    return centroids, clusterAssment, fitness


