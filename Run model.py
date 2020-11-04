# From this tutorial https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
# If I wanna train on ec2 with a GPU, could do this (havnt tested) https://medium.com/coinmonks/a-step-by-step-guide-to-set-up-an-aws-ec2-for-deep-learning-8f1b96bf7984

# I'm using a different dataset (breast cancer) per the readme file

# NOTE need to run in the tensorflow environment, which doesn't work with spyder. So can run like this:
    # conda activate tensorflow
    # then cd into this script's dir, then
    # python "read in images and run model.r"

#%% import libs and set params
#import imageio
from PIL import Image
import glob
import numpy as np
import sys
import os

#from matplotlib import pyplot as pl # for visualising images

# Name of folder to pull images from - 'test images' or 'images' (former much smaller) 
imagedir = 'images' 

# downsample training data?
downsample = True

#%% Read in all X (images) data, reading in pos and neg classes separately
print("reading in images\n")

# pixels for transfer learning 
RESIZE = 224 
def readimages(target):
    """Loops thru png images either posivite or negative for cancer and appends to a numpy array
    Parameters:
    target: *string* either "0" or "1", to read in either negative or positive images
    Returns:
    numpy array of images, of shape (numimages, numpixels1, numpixels2, numRGBVals). So (numimages, 50, 50, 3) here
   """
   
   # initiate array
    l=[]
    for im_path in glob.glob("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/*/" + target + "/*.png"):
        
        im = Image.open(im_path)
        print(im_path)
        
        # only add to data if image has correct dimensions
        if im.size == (50,50):
            im = im.resize((RESIZE, RESIZE))
            im = np.array(im)
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
print(str(xneg.shape[0]) + ' out of ' + str(len(glob.glob("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/*/0/*.png"))) + ' negative images retained')
print(str(xpos.shape[0]) + ' out of ' + str(len(glob.glob("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/*/1/*.png"))) + ' positive images retained')

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

#%% Downsample training data if specified. 
downsample = True
if downsample:
    yneg = yneg[:len(ypos)]
    xneg = xneg[:len(ypos)]
    print('\ndownsampling - ASSUMING POS CLASS IS THE ONE TO DOWNSAMPLE\n')
    # NB would be better to randomly downsample, but I'm shuffling later so just taking first n vakues as a shortcut.

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

#%% split into train and test sets - 80% train. This also shuffles

np.random.seed(1203)
ntot = y.shape[0]
indices = np.random.permutation(ntot)
cutoffIndex = round(ntot*0.8)
training_idx, test_idx = indices[:cutoffIndex], indices[cutoffIndex:]
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

(x_train, y_train), (x_val, y_val) = (x_train, y_train), (x_test, y_test)
del x_test, y_test
from keras import layers
from keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

# normalize inputs from 0-255 to 0.0-1.0 - jay edit: np.max(X_train shows this is still appropriate max value)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train = x_train / 255.0
x_val = x_val / 255.0

## one hot encode outputs ***ALSO UPDATE FINAL DENSE LAYER ***
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
num_classes = y_val.shape[1]

#%% flip, zoom images to improve sample size (and improve model regardless?)
BATCH_SIZE = 16

# train_generator = ImageDataGenerator(
#         zoom_range=2,  # set range for random zoom
#         rotation_range = 90,
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=True,  # randomly flip images
#     )

#%% Define model
model = Sequential()
model.add(DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(2, activation='softmax'))
    
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4),
    metrics=['accuracy']
    )

model.summary()

#%% run

# --------------------- WHERE IM UP TO: -----------------------------
# - haven't really documented anything, so still need to do this
# - check if i should be freezing base model, and if so how? Think this is why it's taking ages.
# - check if my selected base model needs a particular normalisation or not, e.g. may be screwing things by doing val/255


print("Fitting model\n")

learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                  verbose=1,factor=0.2, min_lr=1e-7)

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# myhistory = model.fit_generator(  # only needed if using train_generator, which is failing silently (OOM error i think)
#     train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
#     steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
#     epochs=5,
#     validation_data=(x_val, y_val),
#     callbacks=[learn_control, checkpoint]
# )

model.fit(x_train, y_train, 
          validation_data=(x_val, y_val), 
          epochs=5, batch_size=BATCH_SIZE,
          callbacks=[learn_control, checkpoint])

## [to load an earlier epoch]
#from keras.models import load_model
#model = load_model('model.h5')

# Final evaluation of the model
print("Evaluating model\n")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

history
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()

history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()



#%% Follow along with tutorial!
(X_train, y_train), (X_test, y_test) = (x_train, y_train), (x_test, y_test)
del x_test, x_train

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import cifar10
from keras import metrics # for AUC
from keras.callbacks import ModelCheckpoint

print("normalising inputs and one-hot-encoding target\n")

# normalize inputs from 0-255 to 0.0-1.0 - jay edit: np.max(X_train shows this is still appropriate max value)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

## one hot encode outputs ***ALSO UPDATE FINAL DENSE LAYER ***
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]

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
# Choose final layer ***ALSO HASH/UNHASH ONE HOT ENCODE SECTION ABOVE ***
#model.add(Dense(num_classes, activation='softmax')) # use dense(2) or dense(1) as per this https://stackoverflow.com/questions/61095033/output-layer-for-binary-classification-in-keras
model.add(Dense(1, activation='sigmoid'))

epochs =10
batch_size=32 # orgin tut said 64, but other site says 32 best starting point.
optimizer = 'Adam'

model.compile(loss='binary_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])

print(model.summary())

# instantiate checkpoint (so best epoch can be re-loaded later). From here https://medium.com/@italojs/saving-your-weights-for-each-epoch-keras-callbacks-b494d9648202
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

print("Fitting model\n")
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
model.fit(X_train, y_train, validation_split=0.3, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

## [to load an earlier epoch]
#from keras.models import load_model
#model = load_model('model.h5')

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
imagepredict("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/10272/1/10272_idx5_x1651_y951_class1.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/9347/0/9347_idx5_x51_y451_class0.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/16570/1/16570_idx5_x1501_y1101_class1.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/16167/0/16167_idx5_x1801_y951_class0.png")
imagepredict("C:/Users/new user/Documents/Image recognition in python/cancer-image-classification/" + imagedir + "/14192/0/14192_idx5_x301_y1_class0.png")

# end
l = []
for y in range(len(y_test)):
    l.append(y_test[y][1])

# ---- PROBLEM: it's calssifying everything as zeroes --------    












