from dataset import SiameseLandmarksDataset
from model import SiameseNetwork
from trainer import Trainer

import torch

from torchvision import transforms

siamese_landmark_dataset = SiameseLandmarksDataset(csv_file='toy-dataset/toy-dataset.csv',
                                        root_dir='toy-dataset/',
                                        transform=transforms.Compose([
                                                   transforms.ToPILImage(),
                                                   transforms.Resize(256),
                                                   transforms.RandomCrop(244),
                                                   transforms.ToTensor()]))

model = SiameseNetwork()

trainer = Trainer(siamese_landmark_dataset, model=model, model_parameters=model.parameters,
                    batch_size=8, lr=0.001, shuffle=True)

trainer.train(num_epochs=30)  # This should print status
trainer.test()
