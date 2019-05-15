
from dataset import LandmarksDataset
from loss import ContrastiveLoss

import torch
import torchvision

from torch import optim, nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def load_net(num_classes=2):
    model_conv = torchvision.models.resnet18(pretrained=True)

    # Disable params in original model
    for param in model_conv.parameters():
         param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = 2048                # Depends on network architecture
    model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs / 2)), \
                                  nn.Sigmoid(), \
                                  nn.Linear(int(num_ftrs / 2), num_classes))

    # Set model_conv to net
    net = model_conv
    return net, model_conv.fc.parameters()

class Trainer:
    """CNN Trainer with train and validation split support.

    Examples: 
        Trainer can train and test, given both datasets.
            
            trainer = Trainer(train_dataset, test_dataset, model)
            trainer.train(num_epochs=10)
            trainer.test()

    Attributes:


    """
    def __init__(self, dataset, model, model_parameters, batch_size=16,
                    lr=0.001, lrs_step_size=10, lrs_gamma=0.1, shuffle=True):
        self.dataset = dataset
        self.train_loader, self.val_loader = self.split_dataset(dataset, batch_size=batch_size, shuffle=shuffle)

        self.model, self.model_parameters = model, model_parameters

        self.criterion = ContrastiveLoss()   # For Siamese Learning
        self.optimizer = optim.Adam(self.model_parameters, lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrs_step_size, gamma=lrs_gamma)

        self.batch_size = batch_size

    def split_dataset(self, dataset, batch_size=16, validation_split=0.2, shuffle=False, random_seed=42):
        """Splits dataset into train and validation"""
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=1)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=1)
#        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
#                                                   sampler=train_sampler,
#                                                   num_workers=1)
#        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                        sampler=valid_sampler,
#                                                        num_workers=1)
                
        return train_loader, validation_loader

    def train(self, num_epochs):
        """Trains our model with associated hyperparameters (lr, batch_size)

        Training should save metrics and create checkpoints.

        TODOs:
            * Make compatible with Tensorboard
        """
        
        training_loss_plot = []
        training_error_plot = []
        for epoch in range(num_epochs):  # loop over the dataset multiple times
    
            running_loss = 0.0
            training_loss = 0.0
            training_error = 0.0
            for i_batch, sample_batch in enumerate(self.train_loader):
                # get the inputs
                images0, images1, labels = sample_batch['image0'], sample_batch['image1'], sample_batch['label']
       
                # Cast inputs and labels to torch.Variable
                images0, images1 = Variable(images0), Variable(images1)
                labels = Variable(labels)
    
                # zero the parameter gradients
                self.optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs0, outputs1 = self.model(images0, images1)
                loss = self.criterion(outputs0, outputs1, labels)
                loss.backward()
                self.optimizer.step()
    
                ### STATISTICS ###
                # Training loss
                running_loss += loss.data.item()
                training_loss += loss.data.item()
                
                # Training error
                y_pred0 = outputs0.max(1)[1].numpy()
                y_pred1 = outputs1.max(1)[1].numpy()
                print(y_pred0)
                print(y_pred1)
                y_pred = np.abs(y_pred0 - y_pred1)
                print(y_pred)
                y_true = labels.data.numpy()
                print(y_true)
                
                #print("Pred: " + str(y_pred))
                #print("True: " + str(y_true))
                
                training_error += np.sum(y_pred != y_true)
    
                # Print Statistics
                print_every = 2
                if i_batch % print_every == print_every - 1:
                    print('[%d , %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / self.batch_size))
                    running_loss = 0.0
    
            # Print and store statistics
            training_loss = training_loss / len(self.dataset) / 0.8
            training_error = training_error / len(self.dataset) / 0.8
            print('Training_loss: ' + str(training_loss))
            print('Training error: ' + str(training_error))
            training_loss_plot.append(training_loss)
            training_error_plot.append(training_error)

            ## TODO: Calculate validation error with siamese network
            # # Calculate validation error
            # val_error = 0.0
            # for i_batch, sample_batch in enumerate(self.val_loader):
            #     # get the inputs
            #     images0, images1, labels = sample_batch['image0'], sample_batch['image1'], sample_batch['label']
       
            #     # Cast inputs and labels to torch.Variable
            #     images0, images1 = Variable(images0), Variable(images1)
            #     labels = Variable(labels)
    
            #     # forward + backward + optimize
            #     outputs0, outputs1 = self.model(images0, images1)

            #     # Training error
            #     y_pred = outputs.data.max(1)[1].numpy()
            #     y_true = labels.data.numpy()
                
            #     val_error += np.sum(y_pred != y_true)
            # val_error = val_error / len(self.dataset) / 0.2
            # print('Validation error: ' + str(val_error))
            
            ### Hyperparameter adjustment ##
            # Increment scheduler
            self.lr_scheduler.step(training_loss)
        print('Finished Training')
    
if __name__ == "__main__":
    landmark_dataset = LandmarksDataset(csv_file='toy-dataset/toy-dataset.csv',
                                            root_dir='toy-dataset/',
                                            transform=transforms.Compose([
                                                       transforms.ToPILImage(),
                                                       transforms.Resize(256),
                                                       transforms.RandomCrop(244),
                                                       transforms.ToTensor()]))

    model, fc_parameters = load_net(num_classes=2)
    
    trainer = Trainer(landmark_dataset, model=model, model_parameters=fc_parameters,
                        batch_size=1, lr=0.0001, shuffle=True)

    trainer.train(num_epochs=30)  # This should print status
    trainer.test()

