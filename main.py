from dataset import LandmarksDataset, RandomDataset
from model import SiameseNetwork
from trainer import Trainer

import torch
import numpy as np

from torchvision import transforms

#landmark_dataset = LandmarksDataset(csv_file='toy-dataset/toy-dataset.csv',
#                                        root_dir='toy-dataset/',
#                                        transform=transforms.Compose([
#                                                   transforms.ToPILImage(),
#                                                   transforms.Resize(256),
#                                                   transforms.RandomCrop(244),
#                                                   transforms.ToTensor()]))

num_samples_list = [10, 20, 30, 40, 50] 
input_dims_list = [1, 2, 3, 4]
hidden_dims_list = [1, 2, 3, 4]

def find_capacity(input_dims, hidden_dims):
    num_samples = 2

    overfit = True
    while overfit:
        # Check superset of labels
        label_superset = generateSuperset(num_samples=num_samples)
        for labels in label_superset:
            overfit = checkOverfit(num_samples, input_dims, hidden_dims, labels)

            if not overfit: break

        num_samples += 1

    return num_samples - 1

def generateSuperset(num_samples):
    superset = np.zeros((2 ** num_samples, num_samples))
    
    n = 0
    for i in range(2 ** num_samples):
        bitstring = format(n, 'b')
        for j in range(len(bitstring)):
            superset[i][num_samples - j - 1] = bitstring[len(bitstring) - j - 1]
        n += 1
    return superset


def checkOverfit(num_samples, input_dims, hidden_dims, labels, num_tries=3):
    for _ in range(num_tries):
        random_dataset = RandomDataset(num_samples=num_samples, input_dims=input_dims, useLabels=True, labels=labels)
        
        model = SiameseNetwork(input_dims=input_dims, hidden_dims=hidden_dims, doConv=False)
        
        trainer = Trainer(random_dataset, model=model, model_parameters=model.parameters,
                            batch_size=8, lr=1, shuffle=True, doValidation=False)
        
        trainer.train(num_epochs=30)  # This should print status
        
        if trainer.training_error_plot[-1] == 0:
            return True
    return False
                
for input_dims in input_dims_list:
    for hidden_dims in hidden_dims_list:
        print("Capacity of (input_dims={0}, hidden_dims={1}): ".format(input_dims, hidden_dims) \
                + str(find_capacity(input_dims, hidden_dims)))


#for input_dims in input_dims_list:
#    for hidden_dims in hidden_dims_list:
#        num_samples = 1
#        overfit = False
#        while not overfit:
#            for tries in range(20):
#                random_dataset = RandomDataset(num_samples=num_samples, input_dims=input_dims)
#                
#                model = SiameseNetwork(input_dims=input_dims, hidden_dims=hidden_dims, doConv=False)
#                
#                trainer = Trainer(random_dataset, model=model, model_parameters=model.parameters,
#                                    batch_size=8, lr=1, shuffle=True, doValidation=False)
#                
#                trainer.train(num_epochs=30)  # This should print status
#                
#                if trainer.training_error_plot[-1] == 0:
#                    overfit = True
                

#trainer.test()

