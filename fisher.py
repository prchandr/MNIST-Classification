from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from parse_data import loadData, displayImage
import numpy as np
import matplotlib.pyplot as plt

"""
LDA for: [0, 9]
Training: 59 incorrectly classified out of 11872 images. (0.497% error rate)
Testing: 23 incorrectly classified out of 1989 images. (1.156% error rate)

LDA for: [0, 8]
Training: 133 incorrectly classified out of 11774 images. (1.130% error rate)
Testing: 20 incorrectly classified out of 1954 images. (1.024% error rate)

LDA for: [1, 7]
Training: 91 incorrectly classified out of 13007 images. (0.700% error rate)
Testing: 23 incorrectly classified out of 2163 images. (1.063% error rate)
"""

def fisherLDA(trainImages, trainLabels, testImages, testLabels, class1, class2):
	# Conditionally select only the values of the two classes from the training/testing sets
	newTrainImages = trainImages[(trainLabels == class1) | (trainLabels == class2)]
	newTrainLabels = trainLabels[(trainLabels == class1)  | (trainLabels == class2)]
	newTestImages = testImages[(testLabels == class1) | (testLabels == class2)]
	newTestLabels = testLabels[(testLabels == class1)  | (testLabels == class2)]

	# Fit LDA to the two classes
	lda = LinearDiscriminantAnalysis()
	lda.fit(newTrainImages, newTrainLabels)

	# Predict values of training images
	predictedTrain = lda.predict(newTrainImages)  
	trainErrorCount = (newTrainLabels != predictedTrain).sum()
	print("Training: %d incorrectly classified out of %d images. (%4.3f%% error rate)" % (trainErrorCount, newTrainImages.shape[0], trainErrorCount/newTrainImages.shape[0]*100))

	# Predict values of testing images
	predictedTest = lda.predict(newTestImages)
	testErrorCount = (newTestLabels != predictedTest).sum()
	print("Testing: %d incorrectly classified out of %d images. (%4.3f%% error rate)" % (testErrorCount, newTestImages.shape[0], testErrorCount/newTestImages.shape[0]*100))
	
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

	# Perform LDA on each pair in the list
	classPairs = [[0, 9], [0, 8], [1, 7]]
	for classPair in classPairs:
		print("LDA for: " + str(classPair))
		fisherLDA(trainImages, trainLabels, testImages, testLabels, classPair[0], classPair[1])

if __name__ == "__main__":
	main()