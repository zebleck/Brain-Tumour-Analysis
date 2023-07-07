from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import random_split
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from collections import Counter

def loadAndPrepareData(root_dir, batch_size=32, n_classes=None, augment=True, shuffle=True):

    # Define the mean and standard deviation for normalization (same as ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define the image transformation pipeline with augmentation
    transform = transforms.Compose([
        transforms.Resize(512),                           # Resize the image to 512x512 pixels
        transforms.CenterCrop(512),                        # Perform a center crop of size 512x512 pixels
        transforms.ToTensor(),                              # Convert the image to a tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize the image using the specified mean and standard deviation
    ])

    if augment:
        transform.transforms.insert(2, transforms.RandomHorizontalFlip()),
        transform.transforms.insert(2, transforms.RandomRotation(30))

    # Create an instance of the ImageFolderDataset with the specified root directory and transformation
    dataset = ImageFolderDataset(root_dir=root_dir, transform=transform, n_classes=n_classes)

    # Filter out classes with 10 or less samples
    dataset.filter_samples()

    # Shuffle the indices of the dataset
    # indices = list(range(len(dataset)))
    # np.random.shuffle(indices)
    # subset = torch.utils.data.Subset(dataset, indices)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Get the list of indices to sample from 
    targets = [label for _, label in train_dataset]
    class_sample_count = np.bincount(targets)
    weights = (1. / class_sample_count[targets]).tolist()
    
    # Create a WeightedRandomSampler to oversample the minority class
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create the dataloaders
    if shuffle == True:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create dataloaders for training and validation
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset


# Custom dataset class for loading images from a folder directory and providing them as input data for training or evaluation
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_samples = Counter()
        self.n_classes = n_classes
        self.tumor_type_dict = {}

        # Iterate over each class directory in the root directory
        for i, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)

            # Skip non-directory items
            if not os.path.isdir(class_path):
                continue

            self.tumor_type_dict[i] = class_dir

            # Iterate over each image file in the class directory
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                # Skip non-jpg and non-png files
                if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
                    continue

                # Add the image path and corresponding label to the lists
                self.image_paths.append(image_path)
                self.labels.append(i)
                
                self.class_samples[i] += 1

        self.classes = os.listdir(root_dir)

    def filter_samples(self):
        # If n_classes is specified, only keep the top n classes with the most samples
        if self.n_classes is not None:
            most_common_classes = [class_id for class_id, _ in self.class_samples.most_common(self.n_classes)]
            filtered_paths = []
            filtered_labels = []
            filtered_classes = []
            filtered_class_samples = Counter()

            for path, label in zip(self.image_paths, self.labels):
                if label in most_common_classes:
                    filtered_paths.append(path)
                    filtered_labels.append(label)
                    filtered_classes.append(self.classes[label])
                    filtered_class_samples[label] += 1

            self.image_paths = filtered_paths
            self.labels = filtered_labels
            self.classes = filtered_classes
            self.class_samples = filtered_class_samples
            print(self.class_samples)

            # Reindex the labels so they are consistent
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(set(self.labels)))}
            self.labels = [label_mapping[label] for label in self.labels]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image file and convert it to RGB format
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        # Apply the transformation to the image if there is one
        if self.transform:
            image = self.transform(image)

        return image, label
    
def countSamplesPerClass(loader):
    class_counts = {}
    for _, labels in loader:
        for label in labels:
            if label.item() not in class_counts:
                class_counts[label.item()] = 0
            class_counts[label.item()] += 1
    return class_counts