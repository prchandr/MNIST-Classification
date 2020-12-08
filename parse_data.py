import numpy as np
import struct
import matplotlib.pyplot as plt

"""http://yann.lecun.com/exdb/mnist/"""
def loadData(imageFileName, labelFileName):
    # Opens files as bytes
    imageFile = open(imageFileName, "rb")
    labelFile = open(labelFileName, "rb")

    # Parse image file using format described on the site
    magicNum = struct.unpack('>i', imageFile.read(4))[0]
    numImages = struct.unpack('>i', imageFile.read(4))[0]
    numRows = struct.unpack('>i', imageFile.read(4))[0]
    numColumns = struct.unpack('>i', imageFile.read(4))[0]
    imageList = np.fromfile(imageFileName, dtype=np.ubyte, offset=16).reshape((numImages, numRows*numColumns))
    
    # Parse label file using format described on the site
    magicNum = struct.unpack('>i', labelFile.read(4))[0]
    numItems = struct.unpack('>i', labelFile.read(4))[0]
    labelList = np.fromfile(labelFileName, dtype=np.ubyte, offset=8)
    
    # Closes files
    imageFile.close()
    labelFile.close()

    return imageList, labelList

# Displays image at the index given in the image list
def displayImage(imageList, index):
    width = int(np.sqrt(imageList[0].shape[0]))
    imgplot = plt.matshow(imageList[index].reshape((width, width)))
    plt.show()



