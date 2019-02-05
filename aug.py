from imgaug import augmenters as iaa
from keras.preprocessing import image
import numpy as np
from os import listdir
from PIL import Image
total=0

def load_images(directories=[]):
    #create a blank list to store the images in
    imgs=[]
    #loop over the list of directories we have been passed
    for directory in directories:
        #loop over the files that are contained in the directory
        for file in listdir(directory):
            #load the image in the directory directory with the filename file
            img = image.load_img(directory+file, target_size=None)
            #convert it to a numpy array
            img = np.array(img)
            #append to the imgs list
            imgs.append(img)
            #delete the img variable to save RAM
            del img
    return imgs
# define what we are going to do to the images
seq = iaa.Sequential([
      iaa.Multiply((0.5, 1.5)), #multiply each pixel by a random value 0.5-1.5
      iaa.AdditiveGaussianNoise(scale=(0, 0.075*255),per_channel=True), #create rgb noise
      iaa.Crop(percent=(0, 0.3)), #0.3=30% #crop 0-30% of the pixels from each image
      iaa.Fliplr(0.5) #flip half of the images left to right
      ],random_order=True) #apply the aformentioned transforms in a random order

#load images
healthy=np.array(load_images(["images/healthy/"]))
dying=np.array(load_images(["images/dying/"]))

#create 5 sets of augmented healthy images
for i in range(0,5):
    #augement all the healthy images
    healthy_aug = seq.augment_images(healthy)
    #loop over the augmented images
    for img in healthy_aug:
        #turn the numpy array into an image
        im = Image.fromarray(img)
        #save the image
        im.save("augmented/healthy/aug_"+"{}.png".format(str(total).zfill(8)))
        #allows for incremental naming
        total+=1

total=0
for i in range(0,5):
    dying_aug = seq.augment_images(dying)
    for img in dying_aug:
        im = Image.fromarray(img)
        im.save("augmented/dying/aug_"+"{}.png".format(str(total).zfill(8)))
        total+=1
        
