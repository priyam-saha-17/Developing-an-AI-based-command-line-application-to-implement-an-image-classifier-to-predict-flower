import json
import torch
from torchvision import transforms 
from torchvision import models 
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import densenet121
from collections import OrderedDict
from torch import optim

import matplotlib.pyplot as plt
import time
import os

from os import listdir
from PIL import Image
import numpy as np


import argparse

    
    
        
def transform_data(args):
    #data_dir = 'flowers'
    data_dir = args.data_directory
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(250),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        ]
    )

    data_transforms_test_validate = transforms.Compose(
        [
            transforms.Resize(250),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        ]
    )


    # TODO: Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform = data_transforms_train)
    image_datasets_validate = datasets.ImageFolder(valid_dir, transform = data_transforms_test_validate)
    image_datasets_test = datasets.ImageFolder(test_dir, transform = data_transforms_test_validate)


    # TODO: Using the image datasets and the transforms, define the dataloaders
    dataloaders_train = DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
    dataloaders_validate = DataLoader(image_datasets_validate, batch_size = 64, shuffle = True)
    dataloaders_test = DataLoader(image_datasets_test, batch_size = 64, shuffle = True)
        
    #with open('cat_to_name.json', 'r') as f:
        #cat_to_name = json.load(f)
        
    return dataloaders_train, dataloaders_validate, dataloaders_test, image_datasets_train
        
        


def pre_trained_model2(args, image_datasets_train):
    if(args.architechture is None):
        arch = 'vgg16'
    else:
        arch = args.architechture
        
    if (arch == 'vgg16'):
        base_model = vgg16(pretrained = True)
        input_nodes = 25088
        if args.hidden_units is None:
            hidden_units = 2048
        else:
            hidden_units = int(args.hidden_units)
    
    elif (arch == 'densenet121'):
        base_model = densenet121(pretrained = True)
        input_nodes = 1024
        if args.hidden_units is None:
            hidden_units = 256
        else:
            hidden_units = int(args.hidden_units)
    
    for parameter in base_model.features:
        parameter.requires_grad = False
    
    
    
    
    #Replacing the classifier while features remain frozen
    output_nodes = len(image_datasets_train.class_to_idx)
    base_model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_nodes, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, output_nodes)),
                          ('relu', nn.ReLU()),
                          ('output', nn.LogSoftmax(dim = 1))
                          ]))
    
    if args.gpu == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    base_model = base_model.to(device)
    return base_model 


def funct_criterion():
    criterion = nn.NLLLoss()
    return criterion

        
def funct_optimizer(my_model, alpha = 0.001):
    optimizer = optim.Adam(my_model.classifier.parameters(), alpha)
    return optimizer
     

def check_accuracy(args, base_model, dataloaders):
    correct = 0
    total = 0
    if args.gpu == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    with torch.no_grad():
        for data in dataloaders:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = base_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total
        
     
        
def train_model(args, base_model, optimizer, criterion, dataloaders_train, dataloaders_validate, dataloaders_test):
        print("TRAINING STARTED!!!!")
        start = time.time()

        if args.gpu == 'gpu' and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        num_of_iterations = args.epochs
        print_every = 5
        steps = 0

        for i in range(num_of_iterations):
    
            Loss = []
            Validation_accuracy = []
            count = []
            j = 1
    
    
            print("================================================================")
            print("Iteration --> {}/{} ".format(i+1, num_of_iterations), end = '\n')
            print("================================================================")
    
            for inputs, labels in dataloaders_train:
        
                running_loss = 0
                steps += 1
        
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
        
                #Feeding forward
                outputs = base_model.forward(inputs)
                loss = criterion(outputs, labels)
                #Back-propagation
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
        
                with torch.no_grad():
                    if steps % print_every == 0:
                        validation_accuracy = check_accuracy(args, base_model, dataloaders_validate)
                        print("\t\t")
                        print("Loss : ", round((running_loss/print_every), 4), end = '\t')
                        print("Validation Accuracy : ", round(validation_accuracy, 4))
                        Loss.append(round((running_loss/print_every), 4))
                        Validation_accuracy.append(round(validation_accuracy, 4))
                        count.append(j)
                        j += 1
            """
            print()
            f = plt.figure()
            f.set_figwidth(10)
            f.set_figheight(5)
            plt.plot(count, Loss, label = "Loss")
            plt.plot(count, Validation_accuracy, label = "Validation Accuracy")
            plt.title("Iteration --> {}/{} ".format(i+1, num_of_iterations))
            plt.legend()
            plt.show()
            """
            #torch.cuda.empty_cache()

        
        print("TRAINING OVER!!!!")
        end = time.time()
        print("Training Time: ", round((end - start)/60, 3), " minutes")
        
        test_accuracy(args, base_model, dataloaders_test)
        
        return base_model
        
        
        
def test_accuracy(args, base_model, dataloaders):
    ans = check_accuracy(args, base_model, dataloaders)
    print("Network Accuracy on the Test Dataset = ", round(ans*100, 4), "%")
    
        
        
def save_model(args, base_model, optimizer, image_datasets_train):
        # TODO: Save the checkpoint 
        base_model.class_to_idx = image_datasets_train.class_to_idx

        dictionary = {
                    'transfer model': base_model.cpu(),
                    'input_size': 25088,
                    'output_size': 102,
                    'features': base_model.features,
                    'classifier': base_model.classifier,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': base_model.state_dict(),
                    'idx_to_class': {v:k for k, v in image_datasets_train.class_to_idx.items()}
                    }
        checkpoint = args.save_directory
        torch.save(dictionary, checkpoint)
        
        
        


        
        
def create_model(args):
    
        #print("yes")
        if (args.gpu and not torch.cuda.is_available()):
            raise Exception("GPU option enabled but not detected!!!")
        if (not os.path.isdir(args.data_directory)):
            raise Exception("Directory does not exist!!!")
        data_directory = os.listdir(args.data_directory)
        #if (not set(data_directory).issubset({'train','valid','test'})
            #raise Exception("Missing valid sub-directories!!!")
        if args.architechture not in ('vgg16', 'densenet121', None):
            raise Exception("Wrong CNN architechture chosen!!!")
            
        dataloaders_train, dataloaders_validate, dataloaders_test, image_datasets_train = transform_data(args)
        base_model = pre_trained_model2(args, image_datasets_train)
        criterion = funct_criterion()
        alpha = args.learning_rate
        optimizer = funct_optimizer(base_model, alpha)
        base_model = train_model(args, base_model, optimizer, criterion, dataloaders_train, dataloaders_validate, dataloaders_test)
        save_model(args, base_model, optimizer, image_datasets_train)

    
        
def parse():
        parser = argparse.ArgumentParser(
            description = 'Command-Line Arguments-Parser for training the Neural Network'
        )
        
        parser.add_argument('--data_directory', help = 'data directory(required)', default="./flowers/")
        parser.add_argument('--save_directory', help = 'directory to save a Neural Network', default="./checkpoint.pth")
        parser.add_argument('--architechture', help = 'CNN model to be used for training [vgg16 or densenet121] ', default="vgg16")
        parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
        parser.add_argument('--epochs', action = "store", type = int, default = 5)
        parser.add_argument('--hidden_units', action = "store", default=512)
        parser.add_argument('--gpu', action="store", default="gpu")
        arguments = parser.parse_args()
        return arguments
        
        
        
def main():
        print("Creating a Deep Learning Model based on vgg16 or densenet121 to classify flowers")
        #global args
        args = parse()
        create_model(args)
        print("Finished!!!")
      
     
main()
