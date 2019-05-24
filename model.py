import torchvision
from torchvision import datasets, models, transforms
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self, input_dims=2048, hidden_dims=1, activation_func=nn.Sigmoid(), doConv=True):
        super(SiameseNetwork, self).__init__()
        model_conv = torchvision.models.resnet18(pretrained=True)

        # Disable params in original model
        for param in model_conv.parameters():
             param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        if doConv:
            model_conv.fc = nn.Sequential(nn.Linear(input_dims, int(input_dims / 2)), \
                                      nn.ReLU())
        else: 
            model_conv.fc = nn.Sequential(nn.Linear(input_dims, hidden_dims), activation_func, \
                                          nn.Linear(hidden_dims, 2), activation_func)

        # Set model_conv to net
        self.model = model_conv
        self.parameters = model_conv.fc.parameters()
        
        # Set flags
        self.doConv = doConv   # Determines whether to convolve on input
        
    def single_forward(self, x):
        if self.doConv:
            x = self.model(x)
        else:
            x = self.model.fc(x)
        return x
        
    def forward(self, x0, x1):
        x0 = self.single_forward(x0)
        x1 = self.single_forward(x1)
        return x0, x1

if __name__ == "__main__":
    net = SiameseNetwork()
    print(net)
