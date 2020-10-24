# From this tutorial https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
# If I wanna train on ec2 with a GPU, could do this (havnt tested) https://medium.com/coinmonks/a-step-by-step-guide-to-set-up-an-aws-ec2-for-deep-learning-8f1b96bf7984

# I'm using a different dataset (breast cancer) per the readme file

# NOTE need to run in the tensorflow environment, which doesn't work with spyder. So can run like this:
    # conda activate tensorflow
    # then cd into this script's dir, then
    # python "read in images and run model.r"

#%% import libs
import imageio
import glob
import numpy as np
import sys
#from matplotlib import pyplot as pl # for visualising images

#%% Read in all X (images) data, reading in pos and neg classes separately
print("reading in images\n")

def readimages(target):
    """Loops thru png images either posivite or negative for cancer and appends to a numpy array
    Parameters:
    target: *string* either "0" or "1", to read in either negative or positive images
    Returns:
    numpy array of images, of shape (numimages, numpixels1, numpixels2, numRGBVals). So (numimages, 50, 50, 3) here
   """
    
   # initiate array
    l=[]
    for im_path in glob.glob("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/test images/*/" + target + "/*.png"):
        im = imageio.imread(im_path)
        print(im_path)
        
#        # only add to data if image has correct dimensions
        if im.shape == (50,50,3):
            l.append(im)
    
    # convert to numpy array - necessary as imageio is a nparray class
    l = np.asarray(l)
    return(l)


# create data - xpos and xneg correspond to positive and negative targets
xneg = readimages("0")
xpos = readimages("1")

# check format matches that in tutorial i'm running off
xneg.shape, xpos.shape
xneg.dtype, xpos.dtype

# Report how many images didn't meet shapre requirements *in future need way to interpolate these?*
print(str(xneg.shape[0]) + ' out of ' + str(len(glob.glob("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/test images/*/0/*.png"))) + ' negative images retained')
print(str(xpos.shape[0]) + ' out of ' + str(len(glob.glob("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/test images/*/1/*.png"))) + ' positive images retained')

#%% Create y values (target labels) from image data
print("creating image labels - pos or neg\n")

def createTargets(target):
    """
    Takes 0 or 1 as a string and returns an numpy uint8 list of the corresponding numsamples of either xneg or xpos
    """
    if target == '0':
        targetlength = np.shape(xneg)[0]
        res = np.repeat(0, targetlength)
    elif target == '1':
        targetlength = np.shape(xpos)[0]
        res = np.repeat(1, targetlength)
    else:
        sys.exit('error in defining target')
        
    #convert to uint8
    res = res.astype(np.uint8)
    return(res)
    

# create and check
ypos = createTargets('1')
yneg = createTargets('0')
yneg.dtype, ypos.dtype
yneg.shape, ypos.shape

#%% bind neg and pos values together

print("binding pos and neg values togther, then creating train/test split\n")
x = np.vstack((xpos, xneg)) # stack using vstack - must have same shape for all but first axis
y = np.concatenate((ypos, yneg))

# reshape y from a vector into a 1D array. (May just be symantics but wanna keep in same format as tutorial)
y.shape[0]
y = y.reshape(y.shape[0], 1)

# check 
x.shape[0] == xneg.shape[0] + xpos.shape[0]
y.shape[0] == yneg.shape[0] + ypos.shape[0]

#%% split into train and test sets - 80% train

np.random.seed(1203)
ntot = y.shape[0]
indices = np.random.permutation(ntot)
cutoffIndex = round(ntot*0.8)
training_idx, test_idx = indices[:cutoffIndex], indices[cutoffIndex:]
training, test = x[training_idx,:], x[test_idx,:]
x_train = x[training_idx, :,:,]
x_test = x[test_idx, :,:,]
y_train = y[training_idx]
y_test = y[test_idx]

#  Check
y_train.shape, y_test.shape
x_train.shape, x_test.shape
y_train[0:19]
y_test[0:19]

##%% Check clkassification worked - since only working iwth a few images, manually find those taht are 1s and 0's and check they are in the 1/0 input folders
#y_train
#y_test
#
## pos:
#pl.imshow(x_train[0])
#pl.imshow(x_train[2])
#pl.imshow(x_test[0])
#
## neg
#pl.imshow(x_train[1])
#pl.imshow(x_train[4])
#pl.imshow(x_test[2])

#%% Follow along with tutorial!
(X_train, y_train), (X_test, y_test) = (x_train, y_train), (x_test, y_test)
del x_test, x_train

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import cifar10

print("normalising inputs and one-hot-encoding target\n")

# normalize inputs from 0-255 to 0.0-1.0 - jay edit: np.max(X_train shows this is still appropriate max value)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
print("Creating model\n")
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

epochs = 25
optimizer = 'Adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

print("Fitting model\n")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# Final evaluation of the model
print("Evaluating model\n")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# And how about evaluating a single prediction?
def imagepredict(path):
    
    im = imageio.imread(path)
    im = np.expand_dims(im, axis=0) # cos trained in batchs, input is a tensor of shape [batch_size, image_width, image_height, number_of_channels]. So need to add a batch size dimension like this
    
    # normalise as per training data
    im = im.astype('float32')
    im = im / 255.0
    print(path)
    print(model.predict_classes(im))
    
xx = model.predict_classes(X_test)

# Run prediction for multiple pos and neg images
imagepredict("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/images/10272/1/10272_idx5_x1651_y951_class1.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/images/9347/0/9347_idx5_x51_y451_class0.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/images/16570/1/16570_idx5_x1501_y1101_class1.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/images/16167/0/16167_idx5_x1801_y951_class0.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/breast cancer image classification/images/14192/0/14192_idx5_x301_y1_class0.png")

# end
l = []
for y in range(len(y_test)):
    l.append(y_test[y][1])

# ---- PROBLEM: it's calssifying everything as zeroes --------    











