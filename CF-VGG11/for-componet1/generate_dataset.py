import os
from pickletools import uint8
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from torchvision import transforms
from numpy import linspace

def generateCoordChannel(row, col):
    '''
        0 --> 1
        . ... 1
        ↓ ... 1
        0 ... 1
    '''
    a = linspace(0, 1, col)
    b = a.T
    for i in range(row - 1):
        b = np.c_[b, a]
    channel1 = b.T

    a = linspace(0, 1, row)
    b = a.T
    for i in range(col - 1):
        b = np.c_[b, a]
    channel2 = b
    # channel1代表x方向上的变化, channel2代表y方向上的变化
    return channel1, channel2


'''
    原始数据集需要添加XY坐标信息
    但是标签不需要添加XY坐标信息
'''


# 要生成的图像大小
IMAGE_SIZE = (224, 224)
LABEL_SIZE = (60, 50)

fu_sample = os.listdir('./data/csv/fu/sample')
fu_label = os.listdir('./data/csv/fu/label')

fish_sample = os.listdir('./data/csv/fish/sample')
fish_label = os.listdir('./data/csv/fish/label')

portsur_sample = os.listdir('./data/csv/portsur/sample')
portsur_label = os.listdir('./data/csv/portsur/label')

pure_sample = os.listdir('./data/csv/pure/sample')
pure_label = os.listdir('./data/csv/pure/label')


# 4个循环先把标签图像生成
for comp in fish_label:
    df = pd.read_csv('./data/csv/fish/label/' + comp, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    channel1 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel2 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel3 = data * 255
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')
    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(LABEL_SIZE)(image)
    image.save('./data/label/fish_' + comp.split('.')[0] + '.jpg')

for comp in fu_label:
    df = pd.read_csv('./data/csv/fu/label/' + comp, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    channel1 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel2 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel3 = data * 255
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')
    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(LABEL_SIZE)(image)
    image.save('./data/label/fu_' + comp.split('.')[0] + '.jpg')

for comp in portsur_label:
    df = pd.read_csv('./data/csv/portsur/label/' + comp, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    channel1 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel2 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel3 = data * 255
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')
    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(LABEL_SIZE)(image)
    image.save('./data/label/portsur_' + comp.split('.')[0] + '.jpg')

for comp in pure_label:
    df = pd.read_csv('./data/csv/pure/label/' + comp, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    channel1 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel2 = np.zeros((row_num, col_num)) # 1,2通道不需要再乘255 因为是0
    channel3 = data * 255
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')
    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(LABEL_SIZE)(image)
    image.save('./data/label/pure_' + comp.split('.')[0] + '.jpg')


#=================================================================================#

# 接下来把sample转换成图像
for sample in pure_sample:
    df = pd.read_csv('./data/csv/pure/sample/' + sample, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    # 不带有坐标信息的1 2 通道
    channel1 = np.zeros((row_num, col_num))
    channel2 = np.zeros((row_num, col_num))

    # 带有坐标信息的1 2 通道
    # channel1_, channel2_ = generateCoordChannel(row_num, col_num)
    
    channel3 = data * 255

    # 不带有位置信息
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')

    # 带有位置信息
    # gray_r_ = Image.fromarray(channel1_).convert('L')
    # gray_g_ = Image.fromarray(channel2_).convert('L')

    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(IMAGE_SIZE)(image)
    image.save('./data/train/pure/pure_' + (sample.split('.')[0]).replace(' ','') + '.jpg')

    # image_ = Image.merge('RGB', (gray_r_, gray_g_, gray_b))
    # image_ = transforms.Resize(IMAGE_SIZE)(image_)
    # image_.save('./data/train/pure/pure_' + (sample.split('.')[0]).replace(' ','') + '_coord.jpg')

for sample in fu_sample:
    df = pd.read_csv('./data/csv/fu/sample/' + sample, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    # 不带有坐标信息的1 2 通道
    channel1 = np.zeros((row_num, col_num))
    channel2 = np.zeros((row_num, col_num))

    # 带有坐标信息的1 2 通道
    # channel1_, channel2_ = generateCoordChannel(row_num, col_num)
    
    channel3 = data * 255

    # 不带有位置信息
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')

    # 带有位置信息
    # gray_r_ = Image.fromarray(channel1_).convert('L')
    # gray_g_ = Image.fromarray(channel2_).convert('L')

    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(IMAGE_SIZE)(image)
    image.save('./data/train/fu/fu_' + (sample.split('.')[0]).replace(' ','') + '.jpg')

    # image_ = Image.merge('RGB', (gray_r_, gray_g_, gray_b))
    # image_ = transforms.Resize(IMAGE_SIZE)(image_)
    # image_.save('./data/train/pure/pure_' + (sample.split('.')[0]).replace(' ','') + '_coord.jpg')

for sample in portsur_sample:
    df = pd.read_csv('./data/csv/portsur/sample/' + sample, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    # 不带有坐标信息的1 2 通道
    channel1 = np.zeros((row_num, col_num))
    channel2 = np.zeros((row_num, col_num))

    # 带有坐标信息的1 2 通道
    # channel1_, channel2_ = generateCoordChannel(row_num, col_num)
    
    channel3 = data * 255

    # 不带有位置信息
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')

    # 带有位置信息
    # gray_r_ = Image.fromarray(channel1_).convert('L')
    # gray_g_ = Image.fromarray(channel2_).convert('L')

    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(IMAGE_SIZE)(image)
    image.save('./data/train/portsur/portsur_' + (sample.split('.')[0]).replace(' ','') + '.jpg')

    # image_ = Image.merge('RGB', (gray_r_, gray_g_, gray_b))
    # image_ = transforms.Resize(IMAGE_SIZE)(image_)
    # image_.save('./data/train/pure/pure_' + (sample.split('.')[0]).replace(' ','') + '_coord.jpg')

for sample in fish_sample:
    df = pd.read_csv('./data/csv/fish/sample/' + sample, header=None, index_col=None)
    data = np.array(df)
    # data即fl数据, 需要对fl数据进行归一化处理, 然后再转化为图像数据
    data = (((data[1:]).T)[1:]).T
    # 使用max-min方法对数据进行标准化处理使其区间到0-1, x' = (x - xmin) / (xmax -xmin)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    row_num = (data.shape)[0]
    col_num = (data.shape)[1]
    # 不带有坐标信息的1 2 通道
    channel1 = np.zeros((row_num, col_num))
    channel2 = np.zeros((row_num, col_num))

    # 带有坐标信息的1 2 通道
    # channel1_, channel2_ = generateCoordChannel(row_num, col_num)
    
    channel3 = data * 255

    # 不带有位置信息
    gray_r = Image.fromarray(channel1).convert('L')
    gray_g = Image.fromarray(channel2).convert('L')

    # 带有位置信息
    # gray_r_ = Image.fromarray(channel1_).convert('L')
    # gray_g_ = Image.fromarray(channel2_).convert('L')

    gray_b = Image.fromarray(channel3).convert('L')

    image = Image.merge('RGB', (gray_r, gray_g, gray_b))
    image = transforms.Resize(IMAGE_SIZE)(image)
    image.save('./data/train/fish/fish_' + (sample.split('.')[0]).replace(' ','') + '.jpg')

    # image_ = Image.merge('RGB', (gray_r_, gray_g_, gray_b))
    # image_ = transforms.Resize(IMAGE_SIZE)(image_)
    # image_.save('./data/train/pure/pure_' + (sample.split('.')[0]).replace(' ','') + '_coord.jpg')