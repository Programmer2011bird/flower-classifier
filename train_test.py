from torch import inference_mode, argmax, load, device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Any
import torch.nn as nn


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

def Visualize_test_Model(MODEL:nn.Module, DATALOADER:DataLoader, CLASS_NAMES:list[str], FILE: str):
    MODEL.load_state_dict(load(f"{FILE}", weights_only=True, map_location=device('cpu')))
    MODEL.eval()

    with inference_mode():
        img, label = next(iter(DATALOADER))

        Y_PRED = MODEL(img)
        Y_PRED = argmax(Y_PRED, 1)

        print(Y_PRED)
        print(label)

        img = img.cpu().numpy()
        label = label.cpu().numpy()
        
        for index in range(len(label)):
            plt.imshow(img[index].T)
        
            plt.title(f"{CLASS_NAMES[label[index]]} | {CLASS_NAMES[Y_PRED[index]]}")
            plt.show()
