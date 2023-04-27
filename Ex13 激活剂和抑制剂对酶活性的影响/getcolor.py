import cv2
import numpy as np
from PIL import Image
from  colormath.color_objects import sRGBColor,XYZColor 
from  colormath.color_conversions import convert_color 

data_path = {'AR':['co2\AR1.png','co2\AR2.png','co2\AR3.png','co2\AR4.png'],
        'ph':['co2\pH1.png','co2\pH2.png','co2\pH3.png'],
        'T':['co2\T1.png','co2\T2.png','co2\T3.png']}

data_RGB = {'AR':[],'T':[],'ph':[]}

for group in ('AR','T','ph'):
    for path in data_path[group]:
        image = Image.open(path)
        
        # 要提取的主要颜色数量
        num_colors = 1
        small_image = image.resize((80, 80))
        result = small_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)   # image with 5 dominating colors
        
        result = result.convert('RGB')
        # result.show() # 显示图像
        main_colors = result.getcolors(80*80)[0][1]
        data_RGB[group].append(main_colors)

        '''
        # 显示提取的主要颜色
        for count, col in main_colors:
            if count < 40:
                continue
            a = np.zeros((224,224,3))
            a = a + np.array(col)
            # print(a)
            cv2.imshow('a',a.astype(np.uint8)[:,:,::-1])
            cv2.waitKey()'''
print(data_RGB)

for group in ('AR','T','ph'):
    for i in range(len(data_RGB[group]-1)):
        color1 = data_RGB[group][i]
        for j in range(i,len(data_RGB[group])):
            color2 = data_RGB[group][j]
            pass