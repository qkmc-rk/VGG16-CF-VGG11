import numpy as np
import torch
from FVGG11 import FVGG11
from SimpleCNN import SimpleCNN
import os
from PIL import Image
from torchvision import transforms
from dataloader import FeemDataSet
from torch.utils.data import DataLoader
import pandas as pd
WEIGHT_PATH = r'./logs/FVGG11_Epoch_031_loss_0.0001.pth'

# 填写FVGG16 SimpleCNN FAlexNet
MODEL = 'FVGG11'

if __name__ == '__main__':

    model = FVGG11()
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.eval()

    files = os.listdir('./data/predict')
    datas = []
    # 读入训练集
    with open('cls_predict.txt', "r") as f:
        lines = f.readlines()


    dataset = FeemDataSet(lines, transform=transforms.Compose([transforms.ToTensor()]))
    datas = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (data,target) in enumerate(datas):
        print('第%d张图象...' % (idx + 1))
        pred = model(data)
        pred = torch.detach(pred)
        pred = np.array(pred)
        pred = pred.reshape((60,60))
        pd.DataFrame(pred).to_csv('./predict_result/' + model.name + '_' + str(idx) + '.csv', header=None, index=None)
        pred = pred * 255
        image = Image.fromarray(pred)
        if image.mode == "F":
            image = image.convert('RGB') 
        image.save('./predict_result/' + model.name + '_' + str(idx) + '.jpg')
        # pd.DataFrame(pred).to_csv('./predict_result/' + model.name + '_' + str(idx) + '.csv', header=None, index=None)
