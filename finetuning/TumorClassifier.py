import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


# Define the CNN model
class TumorClassifier(nn.Module):
  def __init__(self, output_dim):
    super(TumorClassifier, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 64 * 64, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, output_dim)

  def forward(self, x):
    batch_size = x.size(0)
    x = nn.functional.relu(self.conv1(x))
    #print('Shape after first conv+relu:', x.shape)
    x = self.pool(nn.functional.relu(self.conv2(x)))
    #print('Shape after second conv+relu+pool:', x.shape)
    x = self.pool(nn.functional.relu(self.conv3(x)))
    #print('Shape after third conv+relu+pool:', x.shape)
    x = self.pool(nn.functional.relu(self.conv4(x)))
    #print('Shape after fourth conv+relu+pool:', x.shape)
    x = x.view(batch_size, -1)
    #print('Shape after flattening:', x.shape)
    x = nn.functional.relu(self.fc1(x))
    #print('Shape after first fc+relu:', x.shape)
    x = nn.functional.relu(self.fc2(x))
    #print('Shape after second fc+relu:', x.shape)
    x = self.fc3(x)
    #print('Shape after final fc:', x.shape)
    return x