from dataset import LandmarksDataset, RandomDataset
from model import SiameseNetwork
from trainer import Trainer

import torch

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


for input_dims in input_dims_list:
    for hidden_dims in hidden_dims_list:
        for num_samples in num_samples_list:
            random_dataset = RandomDataset(num_samples=num_samples, input_dims=input_dims)
            
            model = SiameseNetwork(input_dims=input_dims, hidden_dims=hidden_dims, doConv=False)
            
            trainer = Trainer(random_dataset, model=model, model_parameters=model.parameters,
                                batch_size=8, lr=1, shuffle=True, doValidation=False)
            
            trainer.train(num_epochs=30)  # This should print status
#trainer.test()
