
GENERAL STUFF -------------------------------------------------

aim is to use deep learning code here https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/ to classify cancer data here https://www.kaggle.com/paultimothymooney/breast-histopathology-images?

could also do as per this full tutorial, but wanna try myself. https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/
...or this tutorial https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9

For downloading data direct to EC2: used the kaggle API as described here https://www.kaggle.com/general/6604. notes:
- also had to download a kaggle.json file locally from the kaggle home page and then scp this into the ~/.kaggle folder to get kaggle working (did kaggle --version to check)
- then this command kaggle datasets download paultimothymooney/breast-histopathology-images *into images/ subdir*

RUNNING CODE ON EC2 MACHINE -------------------------------------------------

# ssh into jay-tensorflow2 machine.  *** While in dir with pem file: *** 
ssh -i jay-tensorflow-linux2.pem ubuntu@ec2-public-dns (changes every time?)
# e.g. ssh -i jay-tensorflow-linux2.pem ubuntu@ec2-3-25-106-142.ap-southeast-2.compute.amazonaws.com

# Create directory for project via git
git clone https://github.com/jayruffell/cancer-image-classification.git
cd cancer-image-classification

# download kaggle & move .json file to dir  *** when in cacner dir***
pip install kaggle
cp kaggle.json ~/.kaggle/kaggle.json
kaggle --version # to check

# download images to correct location (after installing kaggle as above)
mkdir images
cd images
kaggle datasets download paultimothymooney/breast-histopathology-images
unzip breast-histopathology-images.zip
rm -r breast-histopathology-images.zip

# create test images (cancer-image-classification dir)
mkdir testimages
cp -R ./images/9173 ./testimages/9173
cp -R ./images/14304 ./testimages/14304

# NB use vi 'Run model.py' to manually set testimages or images dir as input - testimages worked well and proz way quicker (thanks transfer learning)

# download required packages
pip install keras
pip install tensorflow
# then if cudnn error, fix using cudnnenv as described here https://stackoverflow.com/questions/49960132/cudnn-library-compatibility-error-after-loading-model-weights/62610399#62610399. I used cudnnenv to switch to 'v7.6.5-cuda102'

#run script (back in main working dir)
python -i 'Run model.py' # -i lets u keep python vars after script runs





