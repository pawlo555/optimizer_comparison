import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class PlaneDataset(Dataset):
    """We want to split dataset into 80/20 retaining class proportions"""

    def __init__(self, root_dir, train=True, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            train (bool): Weather it is train or test dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        self.image_folder = ImageFolder(
                root_dir,
                transform,
            )

        self.train = train
        self.train_idx = []
        self.test_idx = []
        for i in range(len(self.image_folder)):
            if i % 5 == 0:
                self.test_idx.append(i)
            else:
                self.train_idx.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def __getitem__(self, idx):
        if self.train:
            return self.image_folder[self.train_idx[idx]]
        else:
            return self.image_folder[self.test_idx[idx]]

    @property
    def classes(self):
        return self.image_folder.classes


if __name__ == '__main__':
    preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
    train_dataset = PlaneDataset(
                "../plane_data",
                True,
                preprocessing)
    test_dataset = PlaneDataset(
                "../plane_data",
                False,
                preprocessing)
    print(len(train_dataset), len(test_dataset))
    print(train_dataset[len(train_dataset)-1], test_dataset[len(test_dataset)-1])
    print("Hello")
