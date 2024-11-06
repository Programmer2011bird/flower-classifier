from torch import Tensor, optim, save
from main import CLASS_NAMES
from tqdm import tqdm
import torch.nn as nn
import dataloader
import train_test


EPOCHS: int = 15
LEARNING_RATE: float = 0.001

class cnn_model(nn.Module):
    def __init__(self, input_channel:int, hidden_units:int, output_channels:int) -> None:
        super().__init__()

        self.LAYER1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.LAYER2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*32*32, out_features=output_channels)
        )

    def forward(self, x:Tensor):
        return self.classifier(self.LAYER2(self.LAYER1(x)))


TRAIN_LOADER, TEST_LOADER = dataloader.get_data()
MODEL: cnn_model = cnn_model(3, 10, 102)

LOSS_FN: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
OPTIMIZER: optim.Adam = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

for EPOCH in tqdm(range(EPOCHS)):
    train_loss: int = train_test.train_step(LOSS_FN, OPTIMIZER, MODEL, TRAIN_LOADER)
    test_loss: int = train_test.test_step(LOSS_FN, MODEL, TEST_LOADER)
    
    save(MODEL.state_dict(), "model_cnn.pth")
 
    print(f"TRAIN LOSS: {train_loss}")
    print(f"TEST LOSS: {test_loss}")


train_test.Visualize_test_Model(MODEL, TEST_LOADER, CLASS_NAMES, "model_cnn.pth")
