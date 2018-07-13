from numpy import *
import time
import matplotlib.pyplot as plt
import KMeans

print("step 1: load data")
dataSet = []
fileIn = open("testSet.txt")
for line in fileIn.readlines():
    temp = []
    lineArr = line.strip().split('\t')
    #    temp += [float(x) for x in lineArr]
    s = len(lineArr)
    for j in range(s):
        temp.append(float(lineArr[j]))
        dataSet.append(temp)
fileIn.close()
print("step 2: clustering")
dataSet = mat(dataSet)
k = 15
centroids, clusterAssment, fitness = KMeans.kmeans(dataSet, k)
#print(centroids)
#print(clusterAssment)
#print(DisBtnCluster)
#print(DisInCluster)
print(fitness)

#print ("step 3: show the result..."  )
#KMeans.showCluster(dataSet, k, centroids, clusterAssment)
