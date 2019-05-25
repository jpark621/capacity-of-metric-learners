import torch

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import skimage.io as io

import numpy as np
import matplotlib.pyplot as plt

# Load data
class LandmarksDataset(Dataset):
    """Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Bidict from ids to labels to keep labels within [0, num_classes]
        self.id_to_label = dict()
        self.label_to_id = dict()
        self.num_classes = 0
        

    def __len__(self):
        return len(self.landmarks_metadata)

    def __getitem__(self, idx):
        landmark_id = self.landmarks_metadata['landmark_id'][idx]
        id = self.landmarks_metadata['id'][idx]
        img_name = self.root_dir + str(landmark_id) + "/" + str(id) + ".jpg"
        image = io.imread(img_name)
        
        # If id is not seen, add to id2label bidict
        if landmark_id not in self.id_to_label:
            self.id_to_label[landmark_id] = self.num_classes
            self.label_to_id[self.num_classes] = landmark_id
            self.num_classes += 1
        
        if self.transform:
            image = self.transform(image)
 
        sample = {'image': image, 'label': self.id_to_label[landmark_id]}

        return sample

class RandomDataset(Dataset):
    """Random dataset with input dimensions input_dims."""

    def __init__(self, num_samples=1, input_dims=1, useLabels=False, labels=[]):
        self.input_dims = input_dims

        # Initialize dataset
        self.dataset = np.random.normal(size=(num_samples, input_dims))

        # Initialize labels
        self.labels = labels if useLabels else np.random.randint(0, 2, size=num_samples)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return {'image': torch.FloatTensor(self.dataset[idx]), 'label': torch.from_numpy(np.array(self.labels[idx])).float()}

### SIAMESE DATA SAMPLER ###
class SiameseDataset(Dataset):
    """Landmarks dataset."""

    def __init__(self, dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get two items, with 50% chance similarity/dissimilarity
        landmark_id0, landmark_id1 = None, None
        should_be_similar = np.random.randint(2)
        for i in range(10):
            idx0, idx1 = np.random.choice(len(self.dataset), size=2)

            landmark_id0 = self.dataset[idx0]['label']
            landmark_id1 = self.dataset[idx1]['label']
            
            if (should_be_similar and (landmark_id0 == landmark_id1)): break
            if (not should_be_similar and (landmark_id0 != landmark_id1)): break
 
        # Return sample
        sample = {'image0': self.dataset[idx0]['image'], 'image1': self.dataset[idx1]['image'],
                  'label': torch.from_numpy(np.array(int(landmark_id0 != landmark_id1))).float()}
        return sample

if __name__ == "__main__":
    landmark_dataset = LandmarksDataset(csv_file='small-dataset/small-dataset.csv',
                                       root_dir='small-dataset/',
                                       transform=transforms.Compose([
                                                 transforms.ToPILImage(),
                                                 transforms.Resize(256),
                                                 transforms.RandomCrop(244),
                                                 transforms.ToTensor()]))
    print("Dataset size: " + str(len(landmark_dataset)))
    print("Row 0: " + str(landmark_dataset[0]))

    siamese_landmark_dataset = SiameseDataset(dataset=landmark_dataset)

    sample = next(iter(siamese_landmark_dataset))
    image0, image1, label = sample['image0'], sample['image1'], sample['label']
    plt.imshow(image0.transpose(0, 2).transpose(0, 1))
    plt.show()
    plt.imshow(image1.transpose(0, 2).transpose(0, 1))
    plt.show()
    print(label)

    print(landmark_dataset[0]['label'])

    random_dataset = RandomDataset(input_dims=1)
    print(random_dataset[0]['label'])
