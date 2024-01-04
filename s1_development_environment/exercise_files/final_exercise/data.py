import torch
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    path = '/home/adop/mlops/dtu_mlops/data/corruptmnist/'
    train_images_list, train_labels_list = [], []
    test_images, test_labels = None, None
    for i in range(0,6):

        train_images_path = path + f'train_images_{i}.pt'
        train_labels_path = path + f'train_target_{i}.pt'
        
        train_images = torch.load(train_images_path)
        train_labels = torch.load(train_labels_path)
        
        train_images_list.append(train_images)
        train_labels_list.append(train_labels)

    #Final set of images from the dataset. 
    train_images = torch.cat(train_images_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    test_images = torch.load(path+f'test_images.pt')
    test_labels = torch.load(path+f'test_target.pt')

    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    return train_dataset, test_dataset