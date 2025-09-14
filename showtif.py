import rasterio
import numpy as np
import matplotlib.pyplot as plt


with rasterio.open('EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif') as src:
    red = src.read(4)   # B4 - 红
    green = src.read(3) # B3 - 绿
    blue = src.read(2)  # B2 - 蓝


def normalize(band):
    band_min, band_max = np.percentile(band, (2, 98))
    return np.clip((band - band_min) / (band_max - band_min), 0, 1)

red_norm = normalize(red)
green_norm = normalize(green)
blue_norm = normalize(blue)

# 合并为 RGB 图像
rgb = np.stack([red_norm, green_norm, blue_norm], axis=-1)

# 显示图像
plt.imshow(rgb)
plt.axis('off')
plt.title('RGB from EuroSAT_MS')
plt.show()
