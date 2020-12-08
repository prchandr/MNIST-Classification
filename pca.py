from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from parse_data import loadData, displayImage
import numpy as np
import matplotlib.pyplot as plt
from naive import naiveClassify
from nearest_neighbors import kNeighborsClassify

"""
n:  5
Bayes:
	Training: 21276 incorrectly classified out of 60000 images. (35.460% error rate)
	Testing: 3420 incorrectly classified out of 10000 images. (34.200% error rate)
Nearest Neighbor:
	Training: 11180 incorrectly classified out of 60000 images. (18.633% error rate)
	Testing: 2526 incorrectly classified out of 10000 images. (25.260% error rate)

n:  10
Bayes:
	Training: 13777 incorrectly classified out of 60000 images. (22.962% error rate)
	Testing: 2219 incorrectly classified out of 10000 images. (22.190% error rate)
Nearest Neighbor:
	Training: 2734 incorrectly classified out of 60000 images. (4.557% error rate)
	Testing: 725 incorrectly classified out of 10000 images. (7.250% error rate)

n:  20
Bayes:
	Training: 9545 incorrectly classified out of 60000 images. (15.908% error rate)
	Testing: 1462 incorrectly classified out of 10000 images. (14.620% error rate)
Nearest Neighbor:
	Training: 1133 incorrectly classified out of 60000 images. (1.888% error rate)
	Testing: 303 incorrectly classified out of 10000 images. (3.030% error rate)

n:  50
Bayes:
	Training: 7741 incorrectly classified out of 60000 images. (12.902% error rate)
	Testing: 1234 incorrectly classified out of 10000 images. (12.340% error rate)
Nearest Neighbor:
	Training: 850 incorrectly classified out of 60000 images. (1.417% error rate)
	Testing: 253 incorrectly classified out of 10000 images. (2.530% error rate)

n:  100
Bayes:
	Training: 7864 incorrectly classified out of 60000 images. (13.107% error rate)
	Testing: 1234 incorrectly classified out of 10000 images. (12.340% error rate)
Nearest Neighbor:
	Training: 952 incorrectly classified out of 60000 images. (1.587% error rate)
	Testing: 275 incorrectly classified out of 10000 images. (2.750% error rate)

"""


# File names of the extracted MNIST files
trainImagesFileName = "train-images.idx3-ubyte"
trainLabelsFileName = "train-labels.idx1-ubyte"
testImagesFileName = "t10k-images.idx3-ubyte"
testLabelsFileName = "t10k-labels.idx1-ubyte"

# Loads data into arrays
trainImages, trainLabels = loadData(trainImagesFileName, trainLabelsFileName)
testImages, testLabels = loadData(testImagesFileName, testLabelsFileName)

# Projecting onto 2D
pca = PCA(n_components=2)
pca.fit(trainImages)
proj2D = pca.transform(trainImages)

# Plotting PCA of trainImages onto 2D, color by classes
plt.scatter(proj2D[:,0], proj2D[:,1], c=trainLabels)
plt.title("PCA of Training Data 2D")
plt.show()

# Projecting onto 3D
pca = PCA(n_components=3)
pca.fit(trainImages)
proj3D = pca.transform(trainImages)

# Plotting PCA of trainImages onto 3D, color by classes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj3D[:,0], proj3D[:,1], proj3D[:,2], c=trainLabels)
plt.title("PCA of Training Data 3D")
plt.show()

nValues = [5, 10, 20, 50, 100]
for n in nValues:
	print("\nn: ", n)
	
	# Project onto n-dimensional space	
	pca = PCA(n_components=n)
	pca.fit(trainImages)
	projTrainImages = pca.transform(trainImages)
	projTestImages = pca.transform(testImages)

	# Bayesian Classification
	print("Naive Bayes Classifier:")
	bayesTrainErrorCount, bayesTestErrorCount = naiveClassify(projTrainImages, trainLabels, projTestImages, testLabels)

	# Nearest Neighbors - k=5
	print("Nearest Neighbors Classifier:")
	knnTrainErrorCount, knnTestErrorCount = kNeighborsClassify(projTrainImages, trainLabels, projTestImages, testLabels, 5)