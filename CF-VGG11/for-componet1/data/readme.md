# data文件夹存放数据, csv原始数据, train存放训练集, test存放测试集, predict存放用于预测的图像

# train中按照标签进行分类, test中同样按照标签分类.

# 分类依据：
标签： fu 、 fish、portsur、pure,根据标签读取fu_comp1, fish_comp1, portsur_comp1, pure_comp1等4个拟合对象。

csv目录： 存放原始csv文件, 包括原始数据, 标签拟合数据

label目录： 标签拟合对象

predict： 存放一部分图像用于预测测试的数据，不用进行分类，通过文件名能让用户区分即可。

test： 存放测试集数据

train： 存放训练用数据