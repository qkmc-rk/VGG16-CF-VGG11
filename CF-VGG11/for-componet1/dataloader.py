from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import feem_class
# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
# np.set_printoptions(threshold = np.inf) 

# 若想不以科学计数显示:
# np.set_printoptions(suppress = True)

class FeemDataSet(Dataset):
    def __init__(self, data_pathes, comp = 1, transform = None):
        '''
            comp: 取值 1 - 6, 主要说明这个数据集是为了拟合第几个组分
            data_pathes是要读取的数据集路径数组
            labels和data_pathes长度一致, 用于标明当前数据要拟合的标签是哪个图像
        '''
        # data 和 target 一一对应
        datas = []
        targets = []
        for path in data_pathes:
            path = path.replace('\n', '') # 去掉换行符
            path = path.replace(' ', '') # 去掉空格
            # path = path.replace('\\', '/') # 更改看不惯的杠杆
            data = path.split(';')[1]
            label = path.split(';')[0]
            target_path = './data/label/%s_comp%s.jpg' % (feem_class.classes[int(label)], str(comp))
            datas.append(np.array(Image.open(data).convert('L')))
            targets.append(np.array(Image.open(target_path).convert('L')))
        self.data = np.array(datas)
        self.target = np.array(targets)
        self.transform = transform
    
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        if self.transform:
            data = self.transform(data)
            target = self.transform(target)
        return data, target

    def __len__(self):
        return len(self.data)