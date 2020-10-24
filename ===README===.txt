aim is to use deep learning code here https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/ to classify cancer data here https://www.kaggle.com/paultimothymooney/breast-histopathology-images?

could also do as per this full tutorial, but wanna try myself. https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/

For downloading data direct to EC2: used the kaggle API as described here https://www.kaggle.com/general/6604. notes:
- also had to download a kaggle.json file locally from the kaggle home page and then scp this into the ~/.kaggle folder to get kaggle working (did kaggle --version to check)
- then this command kaggle datasets download paultimothymooney/breast-histopathology-images

# where i'm up to:
- need to re-clone repo into ec2 machien and redownload data *but do into images/ subdir* then unzip.
