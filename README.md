# planthealthcnn

A CNN that uses transfer learning on mobilenet to decide if an image of a plant contains a healthy plant or a dying plant. This is essentially my [catdogcnn](https://github.com/qwertpi/catvdogcnn) code modified to work on 224x224 images of healthy and dying plants. I sourced my training images using the chrome extension [fatkun batch image download](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en) and google images.

![A gif of an image going through the network](viz.gif?raw=true "An image going through the network")

## Copyright 
Copyright © 2019  Rory Sharp All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If you have not received this, see <http://www.gnu.org/licenses/gpl-3.0.html>.

For a summary of the licence go to https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)
## Install and setup
### One liner install
#### Prerequisites
* Curl `sudo apt-get install curl`
* Bash `sudo apt-get install bash`
#### Setup
1. `curl https://raw.githubusercontent.com/qwertpi/catvdogcnn/master/rtinstall.bash | bash`
### Manual install
#### Prerequisites
* Git `sudo apt-get install git`
* [Python 3](https://www.python.org/downloads/)
* ImageMagick (sic) `sudo apt-get install imagemagick`
* libhdf5 (only needed on some systems) `sudo apt-get install libhdf5-serial-dev`

Install the python packages either running `sudo -H pip3 install -U -r requirements.txt` after downloading the repo (after step 0) or installing all the packages named in requirements.txt manually using `sudo -H pip3 install -U package-name`
#### Setup
0\. Download this repo `git clone https://github.com/qwertpi/planthealthcnn.git`  
1\. Source some images of healthy and dying plants and put them in the respective folders in images  
2\. Resize your images `cd images/healthy/ && for file in *; do convert $file -resize 224x224 -gravity center -background "rgb(0,0,0)" -extent 224x224 $file; done` then `cd ..` then `cd dying/ && for file in *; do convert $file -resize 224x224 -gravity center -background "rgb(0,0,0)" -extent 224x224 $file; done` then `cd ..` then another `cd ..`  
3a\. Run aug.py to create new images with added noise, random brighnes and cropping `python3 aug.py`  
3b\. (Optional) Copy your old images to augmented as well `cp images/healthy/* augmented/healthy/ && cp images/dying/* augmented/dying/`  
4\. If you want to run viz.py you will have to replace the keract.py file in wherever your python packeges live with [my forked verison](https://github.com/qwertpi/keract)
## Usage
1\. Run train.py to train the model `python3 train.py`  
2\. (Optional) Run metrics.py to see which images in the training data your model is failing to classify `python3 metrics.py`  
3\. Change the file variable in predict.py to point towards the image you want to check then run it, code for live prediction on webcam feeds may or may not come in the futre  
4\. (Optional) Run viz.py changing the file that is used if you want to see an image go through the network `python3 viz.py`, if you want to make a gif I used [ezgif.com](https://ezgif.com/maker)
## Misc
If you get the /tmp direcotry doesn't use RAM message you can create a 10MB RAM backed directory called /ram by running the command `sudo mkdir -p /ram && sudo mount -t tmpfs -o size=10m tmpfs /ram` and then change [tmp_dir](https://github.com/qwertpi/catvdogcnn/blob/07745d8058cb5fb8e8b346d3023d38c46d80b65d/predict.py#L7) to point to your RAM backed directory eg. tmp_dir="/ram/"  
I have attached 3 images that I believe to be in neither the train nor test set for you to use in prediction testing if you want
