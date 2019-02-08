echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python, the Python Package Installer, software to download code from GitHub and software to resize images'
sudo apt-get install python3 python3-pip git imagemagick -y
echo 'Downloading code from Github'
git clone https://github.com/qwertpi/planthealthcnn
cd planthealthcnn
echo 'Installing the requried python libaries'
sudo pip3 install -r requirements.txt
sudo apt-get install libhdf5-serial-dev
read -p "Please download images and place them in the respective folders in images. When done press enter"
echo 'Resizing images'
cd images/healthy/ && for file in *; do convert $file -resize 224x224 -gravity center -background "rgb(0,0,0)" -extent 224x224 $file; done
cd ..
cd ..
cd images/dying/ && for file in *; do convert $file -resize 224x224 -gravity center -background "rgb(0,0,0)" -extent 224x224 $file; done
cd ..
cd ..
echo 'Augmenting images'
python3 aug.py
echo 'Adding original images to the augmented dataset'
cp images/healthy/* augmented/healthy/ && cp images/dying/* augmented/dying/
echo "You can now run train.py"
