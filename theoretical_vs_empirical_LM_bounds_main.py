from dataset import LandmarksDataset, RandomDataset
from model import SiameseNetwork
from trainer import Trainer

import torch
import numpy as np

from torchvision import transforms

### HELPER FOR FINDING CAPACITIES ###
def find_capacity(input_dims, hidden_dims):
    num_samples = 2

    overfit = True
    while overfit:
        # Check superset of labels
        label_superset = generateSuperset(num_samples=num_samples)
        for labels in label_superset:
            overfit = checkOverfit(num_samples, input_dims, hidden_dims, labels, num_tries=10)

            if not overfit: break

        print("Number of samples: " + str(num_samples))
        num_samples += 1

    return num_samples - 1

def generateSuperset(num_samples, sample_limit=3, LM_not_MK=False):
    if num_samples <= sample_limit:
        num_colorings = 2 ** num_samples if LM_not_MK else 2 ** (num_samples - 1)
        superset = np.zeros((num_colorings, num_samples))
        n = 0
        for i in range(num_colorings):
            bitstring = format(n, 'b')
            for j in range(len(bitstring)):
                superset[i][num_samples - j - 1] = bitstring[len(bitstring) - j - 1]
            n += 1
        return superset
    else:
        num_colorings = 2 ** sample_limit if LM_not_MK else 2 ** (sample_limit - 1)
        superset = np.zeros((num_colorings, num_samples))
        for i in range(num_colorings):
            superset[i] = np.random.randint(0, 2, size=num_samples)
        return superset


def checkOverfit(num_samples, input_dims, hidden_dims, labels, num_tries=3):
    for _ in range(num_tries):
        random_dataset = RandomDataset(num_samples=num_samples, input_dims=input_dims, useLabels=True, labels=labels)
        
        model = SiameseNetwork(input_dims=input_dims, hidden_dims=hidden_dims, doConv=False)
        
        trainer = Trainer(random_dataset, model=model, model_parameters=model.parameters,
                            batch_size=8, lr=1, shuffle=True, doValidation=False)
        
        trainer.train(num_epochs=30)  # This should print status
        
        if trainer.training_error_plot[-1] == 0.0:
            return True
    return False
 
### MAIN SCRIPT ###
if __name__ == "__main__":
    input_dims_list = [1, 2]
    hidden_dims_list = [1, 2, 3, 4]
    
    for input_dims in input_dims_list:
        for hidden_dims in hidden_dims_list:
            print("=== Start finding capacity ===")
            print("Capacity of (input_dims={0}, hidden_dims={1}): ".format(input_dims, hidden_dims) \
                    + str(find_capacity(input_dims, hidden_dims)))
            print("\n")


