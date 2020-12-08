from sklearn.naive_bayes import GaussianNB
from parse_data import loadData, displayImage
import numpy as np

"""
Training: 26106 incorrectly classified out of 60000 images. (43.510% error rate)
Testing: 4442 incorrectly classified out of 10000 images. (44.420% error rate)
"""

def naiveClassify(trainImages, trainLabels, testImages, testLabels):
	# Create a Gaussian Naive Bayes Classifier and fit to training data
	gnb = GaussianNB()
	gnb.fit(trainImages, trainLabels)

	# Predict values of training images
	predictedTrain = gnb.predict(trainImages)  
	trainErrorCount = (trainLabels != predictedTrain).sum()
	print("Training: %d incorrectly classified out of %d images. (%4.3f%% error rate)" % (trainErrorCount, trainImages.shape[0], trainErrorCount/trainImages.shape[0]*100))

	# Predict values of testing images
	predictedTest = gnb.predict(testImages)
	testErrorCount = (testLabels != predictedTest).sum()
	print("Testing: %d incorrectly classified out of %d images. (%4.3f%% error rate)" % (testErrorCount, testImages.shape[0], testErrorCount/testImages.shape[0]*100))
	
	return trainErrorCount, testErrorCount

def main():
	# File names of the extracted MNIST files
	trainImagesFileName = "train-images.idx3-ubyte"
	trainLabelsFileName = "train-labels.idx1-ubyte"
	testImagesFileName = "t10k-images.idx3-ubyte"
	testLabelsFileName = "t10k-labels.idx1-ubyte"

	# Loads data into arrays
	trainImages, trainLabels = loadData(trainImagesFileName, trainLabelsFileName)
	testImages, testLabels = loadData(testImagesFileName, testLabelsFileName)

	# Runs classification on the training and testing data
	naiveClassify(trainImages, trainLabels, testImages, testLabels)

if __name__ == "__main__":
	main()