
GENERAL STUFF -------------------------------------------------

# Originally from this tutorial https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/, which was good for basics but didn't work for the breast cancer images. Ended up doing transfer learning with some help from this tut instead https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9. TL is great when small sample size.

# Prepped code on linux box locally and then ran code on ec2 GPU machine, as described below.

For downloading data direct to EC2: used the kaggle API as described here https://www.kaggle.com/general/6604. notes:
- also had to download a kaggle.json file locally from the kaggle home page and then scp this into the ~/.kaggle folder to get kaggle working (did kaggle --version to check)
- then this command kaggle datasets download paultimothymooney/breast-histopathology-images *into images/ subdir*

RUNNING CODE ON EC2 MACHINE -------------------------------------------------

# ssh into jay-tensorflow2 machine.  *** While in dir with pem file: *** 
ssh -i jay-tensorflow-linux2.pem ubuntu@ec2-public-dns (changes every time?)
# e.g. ssh -i jay-tensorflow-linux2.pem ubuntu@ec2-54-206-122-65.ap-southeast-2.compute.amazonaws.com

-------------- this code only required for initial machine setup --------------------------

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

# download required packages
pip install keras
pip install tensorflow
# then if cudnn error, fix using cudnnenv as described here https://stackoverflow.com/questions/49960132/cudnn-library-compatibility-error-after-loading-model-weights/62610399#62610399. I used cudnnenv to switch to 'v7.6.5-cuda102'
-------------- ------------------------------------------------------------------------------

##run script (back in main working dir)

#first may need to re-install version of cudnn (if seeing cudnn-related error), or at least set environmental vars, as described here https://github.com/unnonouno/cudnnenv. as above i ensured active version was 'v7.6.5-cuda102'. To set env vars, per url:
LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
CPATH=~/.cudnn/active/cuda/include:$CPATH
LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH

# THen run script:
python -i 'Run model.py' # -i lets u keep python vars after script runs



