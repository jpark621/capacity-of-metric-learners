from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import skimage.io as io

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
