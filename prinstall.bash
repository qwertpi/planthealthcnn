echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python, the Python Package Installer, software to download code from GitHub and software to resize images'
sudo apt-get install python3 python3-pip git imagemagick -y
echo 'Downloading code from Github'
git clone https://github.com/qwertpi/catvdogcnn.git catdogcnn
cd catdogcnn
echo 'Installing the required python libraries'
sudo pip3 install -r requirements.txt
sudo apt-get install libhdf5-serial-dev
