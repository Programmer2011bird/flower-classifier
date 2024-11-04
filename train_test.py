from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
# import torch.optim as optim
from typing import Any
import torch.nn as nn
from torch import inference_mode


def train_step(LOSS_FN:Any, OPTIMIZER:Optimizer, MODEL:nn.Module, DATALOADER:DataLoader):
    Train_loss: int = 0
    
    MODEL.train()
    
    for batch, (img, labels) in enumerate(DATALOADER):
        y_preds = MODEL(img)

        OPTIMIZER.zero_grad()
        loss = LOSS_FN(y_preds, labels)
        loss.backward()
        OPTIMIZER.step()

        Train_loss += loss
        
        if batch % 5 == 0:
            print(batch)

    return Train_loss

def test_step(LOSS_FN:Any, MODEL:nn.Module, DATALOADER:DataLoader):
    Test_loss: int = 0
    
    MODEL.eval()
    
    with inference_mode():
        for batch, (img, labels) in enumerate(DATALOADER):
            y_preds = MODEL(img)
            loss = LOSS_FN(y_preds, labels)
            Test_loss += loss

            if batch % 5 == 0:
                print(batch)
    
    return Test_loss
