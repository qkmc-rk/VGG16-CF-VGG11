# 对比插值前后图像的区别变化

# ex: 200 - 550    em:  250 -610

# 未插值之前的数据
originData = 'data/csv/fu/sample/18.csv'

# 插值之后的数据
interpoPic = 'data/test/fu/fu_18.jpg'

import pandas as pd
import numpy as np
from PIL import Image
import os

# 对原始图像作灰度图
df = pd.read_csv(originData)
data = np.array(df)
data = (((data[1:]).T)[1:]).T
# data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
# data = data * 255
image = Image.fromarray(data)
image = image.resize((60, 60))
array = np.array(image)
pd.DataFrame(array).to_csv('./60.csv', header=None,)
# image.convert('L')
# image.save('./image.jpg')