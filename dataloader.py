import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


def get_data(Download_dataset:bool=False, Data_Dir:str="./data", BatchSize:int=32) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ])

    train_dataset = datasets.Flowers102(root=Data_Dir, split="train",
                                        target_transform=None, download=Download_dataset, transform=transform)
    
    test_dataset = datasets.Flowers102(root=Data_Dir, split="test",
                                        target_transform=None, download=Download_dataset, transform=transform)

    train_dataloader: DataLoader = DataLoader(
        dataset=train_dataset, batch_size=BatchSize, shuffle=True)
    
    test_dataloader: DataLoader = DataLoader(
        dataset=test_dataset, batch_size=BatchSize, shuffle=False)
    
    return (train_dataloader, test_dataset)
