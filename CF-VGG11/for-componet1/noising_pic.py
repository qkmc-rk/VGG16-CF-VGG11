import numpy as np
from PIL import Image


class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """
 
    def __init__(self, mean, sigma):
 
        self.mean = mean
        self.sigma = sigma
 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 将图片灰度标准化
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        # 产生高斯 noise
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out*255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('L')