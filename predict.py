from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from subprocess import check_output as bash

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

while True:
    #change this to the file you want to predict on
    #shoudln't have a traling /
    file=input("Enter the filename of the image    ")
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

    #makes the prediction
    prediction=model.predict(img)
    #tells you wether is it healthy or dying and the % certanty
    if prediction.argmax()==0:
        print("Healthy with",str(prediction[0][0]*100)+"%","certianty")
    elif prediction.argmax()==1:
        print("Dying with",str(prediction[0][1]*100)+"%","certianty")
    #this shoudln't ever run but gives me lots of useful debug info
    else:
        print("Well this is awkard. This isn't meant to happen. Please submit an issue on GitHub with the following information")
        print(prediction)
        print(command)
        print(file)
        print(tmp_dir)
        print(bash("df -T "+tmp_dir,shell=True).decode("utf-8"))

    #removes the file from tmp_dir to save RAM
    command="rm "+file
    bash(command)
