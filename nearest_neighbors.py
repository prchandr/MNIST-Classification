from sklearn.neighbors import KNeighborsClassifier
from parse_data import loadData, displayImage
import numpy as np

"""
TRAINING - Took 1 hour to run at 3.6GHz
k = 1:    0.000%
k = 5:    1.808%
k = 10:   2.500%
k = 20:   3.262%
k = 50:   4.637%
k = 100:  5.868%

TESTING - Took 20 minutes to run
k = 1:    3.09%
k = 5:    3.12%
k = 10:   3.35%
k = 20:   3.75%
k = 50:   4.66%
k = 100:  5.60%
"""

def kNeighborsClassify(trainImages, trainLabels, testImages, testLabels, k):
    # Create a k-Nearest-Neighbors classifier and fit to training data
    clf = KNeighborsClassifier(k, n_jobs=-1)
    clf.fit(trainImages, trainLabels)
    
    # Get training error rate
    predictedTrain = clf.predict(trainImages)
    trainErrorCount = (trainLabels != predictedTrain).sum()
    print("[k = %d] Training: %d incorrectly classified out of %d images. (%4.3f%% error rate)" %(k, trainErrorCount, trainImages.shape[0], trainErrorCount/trainImages.shape[0]*100))

    # Get testing error rate
    predictedTest = clf.predict(testImages)
    testErrorCount = (testLabels != predictedTest).sum()
    print("[k = %d] Testing: %d incorrectly classified out of %d images. (%4.3f%% error rate)" % (k, testErrorCount, testImages.shape[0], testErrorCount/testImages.shape[0]*100))
    
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

    # Define list of k for testing, and runs k-Nearest-Neighbors for each value
    kValues = [1, 5, 10, 20, 50, 100]
    for k in kValues:
        kNeighborsClassify(trainImages, trainLabels, testImages, testLabels, k)

if __name__ == "__main__":
    main()