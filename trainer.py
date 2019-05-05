import torch
import torchvision
from torch import optim, nn
from torchvision import datasets, models, transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from dataset import LandmarksDataset

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
                    lr=0.001, lrs_step_size=10, lrs_gamma=0.1):

        self.train_dataset, self.val_dataset = self.split_dataset(dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=8)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=8)

        self.model, self.model_parameters = model, model_parameters

        self.criterion = nn.CrossEntropyLoss()   # For classification tasks
        self.optimizer = optim.Adam(self.model_parameters, lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrs_step_size, gamma=lrs_gamma)

        self.batch_size = batch_size

    def split_dataset(self, dataset):
        """Splits dataset into train and validation"""
        return dataset, dataset  # TODO

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
                inputs, labels = sample_batch['image'], sample_batch['label']
       
                # Cast inputs and labels to torch.Variable
                inputs = Variable(inputs)
                labels = Variable(labels)
    
                # zero the parameter gradients
                self.optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
    
                ### STATISTICS ###
                # Training loss
                running_loss += loss.data.item()
                training_loss += loss.data.item()
                
                # Training error
                y_pred = outputs.data.max(1)[1].numpy()
                y_true = labels.data.numpy()
                
                training_error += np.sum(y_pred != y_true)
    
                # Print Statistics
                print_every = 2
                if i_batch % print_every == print_every - 1:
                    _, y_pred = outputs.data.max(0)
                    y_true = labels.data
    
                    print('[%d , %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / self.batch_size))
                    running_loss = 0.0
    
            # Print and store statistics
            training_loss = training_loss / len(landmark_dataset)
            training_error = training_error / len(landmark_dataset)
            print('Training_loss: ' + str(training_loss))
            print('Training error: ' + str(training_error))
            training_loss_plot.append(training_loss)
            training_error_plot.append(training_error)
            
            # Increment scheduler
            self.lr_scheduler.step(training_loss)
        print('Finished Training')
        
        # Store statistics
        training_loss_plot_list.append(training_loss_plot)
        training_error_plot_list.append(training_error_plot)
    
if __name__ == "__main__":
    landmark_dataset = LandmarksDataset(csv_file='small-dataset/small-dataset.csv',
                                            root_dir='small-dataset/',
                                            transform=transforms.Compose([
                                                       transforms.ToPILImage(),
                                                       transforms.Resize(256),
                                                       transforms.RandomCrop(244),
                                                       transforms.ToTensor()]))

    model, fc_parameters = load_net(num_classes=10)
    
    trainer = Trainer(landmark_dataset, model=model, model_parameters=fc_parameters,
                        batch_size=16, lr=0.001)

    trainer.train(num_epochs=30)  # This should print status
    trainer.test()


