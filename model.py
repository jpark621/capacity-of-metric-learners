import torchvision
from torchvision import datasets, models, transforms
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        model_conv = torchvision.models.resnet18(pretrained=True)

        # Disable params in original model
        for param in model_conv.parameters():
             param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = 2048                # Depends on network architecture
        model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs / 2)), \
                                      nn.ReLU())

        # Set model_conv to net
        self.model = model_conv
        self.parameters = model_conv.fc.parameters()
        
        
    def single_forward(self, x):
        x = self.model(x)
        return x
        
    def forward(self, x0, x1):
        x0 = self.single_forward(x0)
        x1 = self.single_forward(x1)
        return x0, x1

if __name__ == "__main__":
    net = SiameseNetwork()
    print(net)
