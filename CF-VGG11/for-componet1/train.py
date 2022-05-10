from pyexpat import model
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from FAlextNet import FAlexNet
from FVGG11 import FVGG11
from SimpleCNN import SimpleCNN
from dataloader import FeemDataSet
import torch.backends.cudnn as cudnn
from torch import nn
import torch
from FAlextNet import FAlexNet
from FVGG11 import FVGG11
from SimpleCNN import SimpleCNN
from loss_history import LossHistory

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

EPOCH = 40  # 迭代训练20次
# 太大容易出现超调现象，即在极值点两端不断发散，或是剧烈震荡，总之随着迭代次数增大loss没有减小的趋势；
# 太小会导致无法快速地找到好的下降的方向，随着迭代次数增大loss基本不变。
# 学习率越小，损失梯度下降的速度越慢，收敛的时间更长 
LR = 10e-5 # 学习率设置为10e-4

BATCH_SIZE = 8

model = FAlexNet()
#model = FVGG11()
model = SimpleCNN()

# 优化器使用Adam
optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 5e-4)
# stepLR学习率调整器  step_size指epoch的大小, gamma是调整倍数  lr = lr * gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.94)

if __name__ == "__main__":

    cuda = False
    #------------------------------------------------------#
    #   获得图片路径和标签
    #------------------------------------------------------#
    annotation_path = 'cls_train.txt'
    #------------------------------------------------------#
    #   进行训练集和验证集的划分，默认使用10%的数据用于验证
    #------------------------------------------------------#
    val_split       = 0.2
    #----------------------------------------------------#
    #   输入的图片大小
    #----------------------------------------------------#
    input_shape     = [60, 60]
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------#
    num_workers     = 0
    #----------------------------------------------------#
    #   训练过程中保存损失值的对象
    #----------------------------------------------------#
    loss_history = LossHistory("loss_logs/", model.name)
    #----------------------------------------------------#
    #   使用GPU进行训练
    #----------------------------------------------------#
    model_train = model.train() # 如果使用CPU 开启此项 注释下面三项
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    #----------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val
    # 训练集数量
    train_datas = lines[:num_train]
    val_datas = lines[num_train:]
    # dataset
    train_dataset = FeemDataSet(train_datas, transform=transforms.Compose([transforms.ToTensor()]))
    val_dataset = FeemDataSet(val_datas, transform=transforms.Compose([transforms.ToTensor()]))
    
    # 训练数据和验证数据
    train_datas = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_datas = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)


    epoch_step      = num_train // BATCH_SIZE
    epoch_step_val  = num_val // BATCH_SIZE


    # epoch
    for epoch in range(EPOCH):
        total_loss      = 0
        total_accuracy  = 0
        # 验证集损失
        val_loss        = 0
        # 开启训练模式
        model_train.train()
        print('start training...')
        # iteration
        for idx, batch in enumerate(train_datas):
            torch.cuda.empty_cache()
            print('iter %d doing...' % (idx+1))
            images, targets = batch
            optimizer.zero_grad()# 梯度清零
            outputs = model_train(images)
            real_batch_size =  targets.size(0)
            loss_value  = nn.MSELoss()(outputs, targets.view(real_batch_size, -1))
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()
            with torch.no_grad():  # 求准确率
                # 求余弦相似度
                sim = torch.cosine_similarity(outputs, targets.view(real_batch_size, -1))
                cossin_all = 0.
                i = 0
                for p in sim:
                    i += 1
                    cossin_all += p.item()
                accuracy = cossin_all / i
                print('iter', idx+1, '余弦相似度(准确率):', accuracy)
                total_accuracy += accuracy  # 总的准确率
            print('iter %d finished, total_loss:%4f, cossine similarity: %4f' % (idx+1, total_loss / (idx+1), total_accuracy / (idx+1)))
        
        print('Finish Train')
        accuracy_file = open('./logs/%s_train_similarity.txt' % model.name, 'a')
        accuracy_file.write(str(total_accuracy / (idx+1)))
        accuracy_file.write('\n')

        # 在这里eval模型
        model_train.eval() # 开启eval
        print('Start Validation')
        total_val = 0
        positive_val = 0
        for iteration, batch in enumerate(val_datas):
            images, targets = batch
            with torch.no_grad():
                optimizer.zero_grad()
                outputs     = model_train(images)
                real_batch_size =  targets.size(0)
                loss_value  = nn.MSELoss()(outputs, targets.view(real_batch_size, -1))
                val_loss    += loss_value.item()
            print('val total loss:', val_loss / (iteration + 1), ',  lr:', get_lr(optimizer))
            sim = torch.cosine_similarity(outputs, targets.view(real_batch_size, -1))
            for p in sim:
                if p.item() >= 0.9: # 相当于置信度
                    positive_val += 1
                total_val += 1
        val_precision_file = open('./loss_logs/%s_val_precision.txt' % model.name, 'a')
        val_precision_file.write('epoch%3d, validation for 0.9 sim: %5f' % (epoch, float(positive_val) /total_val))
        val_precision_file.write('\n')

        loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
        print('Finish Validation')

        print('epoch %d/%d finished!' % (epoch+1, EPOCH))
        print('Total Loss: %.6f || Val Loss: %.6f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        loss_file = open('./loss_logs/%s_loss-history.txt'  % model.name, 'a')
        loss_file.write(str(total_loss / epoch_step))
        loss_file.write(',')
        loss_file.write(str(val_loss / epoch_step_val))
        if epoch % 10 == 0:
            torch.save(model_train.state_dict(), './logs/%s_Epoch_%03d_loss_%.4f.pth' % (model.name, epoch + 1, loss_value.item()))

        scheduler.step()
    loss_history.loss_plot()




