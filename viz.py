from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from subprocess import check_output as bash
import keract

#in variable to allow this to be changed if /tmp isn't RAM backed but a ramdisk has been created
tmp_dir='/tmp/'
#tells you if your tmp_dir is RAM backed or not
if "tmpfs" in bash("df -T "+tmp_dir,shell=True).decode("utf-8"):
    print("Your system supports writing to RAM so image loading should be fast :)")
else:
    print("Your /tmp directory doesn't use RAM. This isn't an error it just means resizing the image will be a bit slower")

#here as we are redefining bash
from os import system as bash

#loads the model from the saved model file
model = load_model('model.h5')
#change this to the file you want to predict on
#shoudln't have a traling /
file="test2.jpg"
#removes trailing / although I'd rather people didn't rely on this
if file[-1]=="/":
    file[:-1]
#creates the command we will run to resize the image and save it to the tmp_dir for faster acsses
command="convert "+file+' -resize 224x224 -gravity center -background "rgb(0,0,0)" -extent 224x224 '+tmp_dir+file.split("/")[-1]
#redefines file to point towards the file in tmp_dir
file=tmp_dir+file.split("/")[-1]
#status code 0 means everything went OK
if bash(command)!=0:
    print("Sorry but on some systems the commands refuse to execute and give an error code amounting to command too long. Please copy paste the following command into a terminal and then press enter in this window")
    input(command)
    
#loads the image
img = image.load_img(file, target_size=None)
#converts it to a numpy array
img = np.array(img)
#noramlizes it
img=np.array([img/255])

#extarcts the mobilenet model
cnn=model.layers[0]
#compiles the mobilenet model else Keras throws a hissy fit
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#loops over each layer number
for i in range(1,14):
    #gets the acitvations of the layer number
    activations = keract.get_activations(cnn, img,"conv_pw_"+str(i))
    #displays the activations
    keract.display_activations(activations,"gray")
