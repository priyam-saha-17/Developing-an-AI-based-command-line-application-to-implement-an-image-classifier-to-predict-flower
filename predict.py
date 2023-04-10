#imports here
import argparse
import json
import torch
from torchvision import transforms 
from torchvision import models 
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision.models import vgg16
from collections import OrderedDict
from torch import optim

import matplotlib.pyplot as plt
import time

from os import listdir
from PIL import Image
import numpy as np




def load_checkpoint(args):
    #print(type(args))
    checkpoint_path = args.save_directory
    base_model_info = torch.load(checkpoint_path)
    base_model = base_model_info['transfer model']
    
    base_model.classifier = base_model_info['classifier']
    base_model.load_state_dict(base_model_info['state_dict'])
    idx_to_class = base_model_info['idx_to_class']
    return base_model, base_model_info, idx_to_class
    




def process_image(args):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image_path = args.image_input
    image = Image.open(image_path)
    image.load()
    
    
    # Step 1 : Resize image
    
    coordinates = list(image.size)
    short_side = min(coordinates)
    short_side_index = coordinates.index(short_side)
    if short_side_index == 0:
        long_side_index = 1
    else:
        long_side_index = 0
    aspect_ratio = coordinates[long_side_index] / coordinates[short_side_index]
    coordinates[short_side_index] = 256
    coordinates[long_side_index] = int(256 * aspect_ratio)
    resized_image = image.resize(coordinates)
    
    
    
    # Step 2 : Crop image
    
    width, height = resized_image.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    right = (width + new_width)/2
    top = (height - new_height)/2
    bottom = (height + new_height)/2
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    
    processed_image = np.array(cropped_image)
    
    # convert colour channel from 0-255, to 0-1
    
    processed_image = processed_image.astype('float64')
    processed_image = processed_image / 255
    
    # normalize for model
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    

    processed_image = (processed_image - mean) / std
    #print(processed_image.shape)
    
    
    # tranpose color channge to 1st dim
    processed_image = processed_image.transpose((2, 0, 1))
    
    #print(processed_image.shape)
    return processed_image
    
    
    
    
def predict(image_path, args, idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    #model = args.save_directory
    with torch.no_grad():
        image = process_image(args)
        #print(image.shape)
        #print(image)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model,_,_ = load_checkpoint(args)
        #print(image.shape)
        #print(image)
        outputs = model(image)
        probabilities, indices = torch.exp(outputs).topk(args.top_k)
        indices = np.array(indices)
        #print(indices)
        top_classes = [int(idx_to_class[x]) for x in indices[0]]
        
        return probabilities[0].tolist(), top_classes
    
    
def read_cat_to_name(args):
    cat_file = args.category_names
    jfile = json.loads(open(cat_file).read())
    return jfile



    
def sanity_check(args, model, idx_to_class):
    path_string = args.image_input
    probabilities, classes = predict(path_string, args, idx_to_class)
    
    print("For given image, the predicted class values along with their associated probabilities are:")
    print("Class  Probability")
    for i in range(0, len(classes)):
        print(classes[i], "\t", round(probabilities[i],3))
            
    cat_to_name = read_cat_to_name(args) 
    
    
    path_list = path_string.split("/")
    ground_truth = cat_to_name[path_list[3]]
    
    
    flower_classes = []
    for ele in classes:
        flower_classes.append(cat_to_name[str(ele)] + "(" + str(ele) + ")")
    image = Image.open(path_string)
    
    #fig, ax = plt.subplots(2, 1, figsize=(3,7))
    #ax[0].imshow(image)
    #y_positions = list(range(len(flower_classes)))
    
    #ax[1].barh(y_positions, probabilities)
    #ax[1].set_yticks(y_positions)
    #ax[1].set_yticklabels(flower_classes)
    #ax[1].set_xlabel('Probability of class (in %)')
    #ax[1].invert_yaxis()
    #ax[1].set_title('Top 5 Class Predictions')
    #print(flower_classes)
    
    #ax[0].set_title("Actual Flower Name = {} ".format(ground_truth))
    
    
    # Finding the predicted flower name
    predicted_flower = cat_to_name[str(classes[0])]
    
    
    # Printing results
    print("Actual Flower Name = {} ".format(ground_truth))
    print("Predicted Flower Name = {} ".format(predicted_flower), end = ",")
    print("Associated Probability = {} ".format(round(probabilities[0],3)))
    if (ground_truth == predicted_flower):
        print("Correct Prediction")
    elif int(path_list[3]) in classes:
        print("Prediction is incorrect but actual flower name lies in the top 5 predictions")
    else:
        print("Incorrect Prediction" )

          
    
    
def parse():
        parser = argparse.ArgumentParser(
            description = 'Command-Line Arguments-Parser for using the trained Neural Network to predict an image class'
        )
        
        parser.add_argument('--image_input', help = 'image file to predict(required)', default="./flowers/train/1/image_06734.jpg")
        parser.add_argument('--save_directory', help = 'directory to load the neural network from', default="./checkpoint.pth")
        parser.add_argument('--top_k', action="store", type=int,default=5)
        parser.add_argument('--category_names', action = "store", type = str, default = 'cat_to_name.json')
        parser.add_argument('--gpu', action="store", default="gpu")
        arguments = parser.parse_args()
        return arguments
        
        
        
def main():
        print("Using a trained Deep Learning Model based on vgg16 or densenet121 to classify flowers")
        #global args
        args = parse()
        if (args.gpu and not torch.cuda.is_available()):
            raise Exception("GPU option enabled but not detected!!!")
        #print(type(args))
        base_model, base_model_info, idx_to_class = load_checkpoint(args)
        #processed_image = process_image(path_string)
        #path_string = args.image_input
        sanity_check(args, base_model, idx_to_class)
        print("Finished!!!")
      
     
main()