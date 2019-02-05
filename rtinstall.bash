echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python, the Python Package Installer, software to download code from GitHub and software to resize images'
sudo apt-get install python3 python3-pip git imagemagick -y
echo 'Downloading code from Github'
git clone https://github.com/qwertpi/catvdogcnn.git catdogcnn
cd catdogcnn
echo 'Installing the requried python libaries'
sudo pip3 install -r requirements.txt
sudo apt-get install libhdf5-serial-dev
echo 'Auto creating folders to store images in. You can kill this install from this point onwards if you need to modfiy anything'
sleep 1s
mkdir images && mkdir images/cats && mkdir images/dogs
echo 'Downloading images (this might take a while) (error messages are safe to ignore as long as they are not in excess as an image might have just been taken offline)'
sleep 1s
python3 getdata.py -u dogs.txt -o images/dogs/
python3 getdata.py -u cats.txt -o images/cats/
echo 'Resizing images'
sleep 1s
cd images/cats/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done
cd ..
cd ..
cd images/dogs/ && for file in *; do convert $file -resize 64x64 -gravity center -background "rgb(0,0,0)" -extent 64x64 $file; done
cd ..
cd ..
read -p "Modifying keras source code to download the correct weights. This runs the risk of messing up Keras on your system. To continue press enter"
sudo sed -i -e 's/mobilenet_%s_%d/mobilenet_%s_224/g' /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py 
sudo sed -i -e 's/alpha_text, rows/alpha_text/g' /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py 
sudo sed -i -e 's/raise ValueError/warnings.warn/g' /usr/local/lib/python3.5/dist-packages/keras_applications/mobilenet.py 
echo 'Done. You can now run train.py'
