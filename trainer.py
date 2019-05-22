
from dataset import LandmarksDataset, SiameseDataset
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

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA

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
        # Load data
        self.dataset = dataset
        self.train_dataset, self.val_dataset = self.split_dataset(dataset)
        self.siamese_train_dataset, self.siamese_val_dataset = SiameseDataset(self.train_dataset), SiameseDataset(self.val_dataset)
        self.train_loader = self.load_data(self.siamese_train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.val_loader = self.load_data(self.siamese_val_dataset, batch_size=batch_size, shuffle=shuffle)

        # Load model
        self.model, self.model_parameters = model, model_parameters

        self.criterion = ContrastiveLoss()   # For Siamese Learning
        self.optimizer = optim.Adam(self.model_parameters, lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lrs_step_size, gamma=lrs_gamma)

        # Hyperparameters
        self.batch_size = batch_size

    def split_dataset(self, dataset, shuffle_val_split=True, validation_split=0.2, random_seed=42):
        """Splits dataset into train and validation"""
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_val_split:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        return train_dataset, val_dataset

    def load_data(self, dataset, batch_size=16, shuffle=True, num_workers=1):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def train(self, num_epochs):
        """Trains our model with associated hyperparameters (lr, batch_size)

        Training should save metrics and create checkpoints.

        TODOs:
            * Make compatible with Tensorboard
        """
        training_loss_plot = []
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            training_loss = 0.0
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
    
                # Print Statistics
                print_every = 2
                if i_batch % print_every == print_every - 1:
                    print('[%d , %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / self.batch_size))
                    running_loss = 0.0
    
            # Print and store statistics
            training_loss = training_loss / len(self.dataset) / 0.8
            print('Training_loss: ' + str(training_loss))
            training_loss_plot.append(training_loss)

            # Calculate validation error
            validation_loss = 0
            for i_batch, sample_batch in enumerate(self.val_loader):
                # get the inputs
                images0, images1, labels = sample_batch['image0'], sample_batch['image1'], sample_batch['label']
       
                # Cast inputs and labels to torch.Variable
                images0, images1 = Variable(images0), Variable(images1)
                labels = Variable(labels)
    
                # forward + backward + optimize
                outputs0, outputs1 = self.model(images0, images1)

                # Validation loss
                loss = self.criterion(outputs0, outputs1, labels)
                validation_loss += loss.data.item()

            validation_loss = validation_loss / len(self.dataset) / 0.2
            print('Validation loss: ' + str(validation_loss))

            # Calculate knn prediction error
            print('=== knn prediction ===')
            loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset))
            train_dataset = next(iter(loader))
            train_outputs = self.model.single_forward(train_dataset['image'])
            train_outputs = train_outputs.detach().numpy()
            #nca = NeighborhoodComponentsAnalysis(n_components=10, random_state=42)
            #nca.fit(outputs, self.dataset['label'])
            pca = PCA(n_components=2)
            pca.fit(train_outputs)
            knn = KNeighborsClassifier(n_neighbors=2)
            knn.fit(pca.transform(train_outputs), train_dataset['label'])
            y_pred = knn.predict(pca.transform(train_outputs))
            y_true = train_dataset['label'].detach().numpy()
            print("KNN training error: " + str(np.mean(y_pred != y_true)))
            
            loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset))
            val_dataset = next(iter(loader))
            val_outputs = self.model.single_forward(val_dataset['image'])
            val_outputs = val_outputs.detach().numpy()

            y_pred = knn.predict(pca.transform(val_outputs))
            y_true = val_dataset['label'].detach().numpy()
            print("KNN validation error: " + str(np.mean(y_pred != y_true)))

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

