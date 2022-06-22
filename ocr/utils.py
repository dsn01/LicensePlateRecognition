import glob
import os
import sys
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
'''
CCPD:  A Very Large Dataset For License Plate Detection And Recognition
CCPD使用'-'将文件名分为7个部分, 以025-95_113-154 & 383_386 & 473-386 & 473_177 & 454_154 & 383_363 & 402-0_0_22_27_27_33_16-37-15.jpg为例
    1) 025：区域
    2) 95_113：水平倾斜度_垂直倾斜度
    3) 154 & 383_386 & 473：左上(154, 383)_右下(386, 473)
    4) 386 & 473_177 & 454_154 & 383_363 & 402：右下(386, 473)_左下(177, 454)_左上(383, 363)_右上(363, 402)
    5) 0_0_22_27_27_33_16：
            provinces[0]_alphabets[0]_ads[22]_ads[27]_ads[27]_ads[33]_ads[16]
    6) 37：亮度
    7) 15：模糊度

base:
    图片数： 199996
    classes：31
    #base数据集中各标签分布：
        base_dict={'皖': 191811, '沪': 663, '苏': 3311, '云': 17, '豫': 431, '京': 307, '川': 121, '浙': 1350, '赣': 151, '湘': 96, '渝': 74, '鲁': 248, '粤': 373, '晋': 67,
            '冀': 175, '闽': 208, '津': 60, '辽': 59, '鄂': 278, '甘': 18, '新': 16, '贵': 16, '陕': 55, '琼': 9, '桂': 15, '青': 11, '蒙': 19, '黑': 18, '吉': 13, '宁': 5, '藏': 1}
    #训练集中各标签分布：    
        train_dict={'皖': 95894, '沪': 336, '苏': 1655, '云': 6, '豫': 215, '京': 147, '川': 66, '浙': 637, '赣': 79, '湘': 47, '渝': 37, '鲁': 133, '粤': 195, '晋': 31,
            '冀': 92, '闽': 116, '津': 33, '辽': 32, '鄂': 142, '甘': 9, '新': 7, '贵': 11, '陕': 31, '琼': 5, '桂': 8, '青': 4, '蒙': 9, '黑': 13, '吉': 6, '宁': 3, '藏': 1}
    #验证集中各标签分布：
        val_dict={'皖': 95917, '沪': 327, '苏': 1656, '云': 11, '豫': 216, '京': 160, '川': 55, '浙': 713, '赣': 72, '湘': 49, '渝': 37, '鲁': 115, '粤': 178, '晋': 36,
            '冀': 83, '闽': 92, '津': 27, '辽': 27, '鄂': 136, '甘': 9, '新': 9, '贵': 5, '陕': 24, '琼': 4, '桂': 7, '青': 7, '蒙': 10, '黑': 5, '吉': 7, '宁': 2, '藏': 0}

'''

provinces = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁',
             '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学', 'O']
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
my_dict = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁',
    '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

res = {

}
# 从原始数据产生训练和测试标注文件


def analysis_file(filename):
    # 解析图片文件名提取车牌的左上角和右小角坐标
    seven_parts = filename.split('-')
    str_coordinates = seven_parts[3]
    str_indices = seven_parts[4]
    # 处理坐标
    coordinates = []
    x_min = 1e5
    y_min = 1e5
    x_max = 0
    y_max = 0
    str_coordinates = str_coordinates.split('_')
    for single in str_coordinates:
        single = single.split('&')
        x, y = int(single[0]), int(single[1])
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
        coordinates.append((x, y))
    left_top, right_bottom = (x_min, y_min), (x_max, y_max)
    # 处理车牌字符
    LPchars = []
    str_indices = str_indices.split('_')
    LPchars.append(provinces[int(str_indices[0])])
    res[provinces[int(str_indices[0])]] = res.get(
        provinces[int(str_indices[0])], 0) + 1
    LPchars.append(alphabets[int(str_indices[1])])
    for i in range(2, 7):
        LPchars.append(ads[int(str_indices[i])])
    LP = ''.join(LPchars)
    return coordinates, LP, left_top, right_bottom


def generate(data_path):
    data_path = 'E:/BaiduNetdiskDownload/新建文件夹/CCPD2019/ccpd_base'
    dirs = os.listdir(data_path)
    print('图片数：', len(dirs))
    for i, img_name in enumerate(dirs):
        _, LP, left_top, right_bottom = analysis_file(img_name)
        img = Image.open(data_path + '/' + img_name)
        img = img.crop((left_top[0], left_top[1],
                        right_bottom[0], right_bottom[1]))
        img = img.resize((100, 32))
        img.save('./data/images/' + str(i) + '.' + LP + '.jpg')
    print(res)


def dataset_split(path, local_path, img_path, base_path):
    # 用于从原始CCPD的数据集格式中切出车牌统一大小的区域图片,并解码图片名进行标注
    with open(path, 'r', encoding='utf-8') as f_in:
        with open(local_path, 'w', encoding='utf-8') as f_out:
            for idx, line in enumerate(f_in):
                _, LP, left_top, right_bottom = analysis_file(line)
                img = Image.open(base_path + line[:-1])
                img = img.crop((left_top[0], left_top[1],
                                right_bottom[0], right_bottom[1]))
                img = img.resize((100, 32))
                img.save(img_path + str(idx) + '_' + LP + '.jpg')
                f_out.write(img_path+str(idx)+'_' +
                            LP + '.jpg'+';' + LP + '\n')


if __name__ == '__main__':
    train_path = 'E:/BaiduNetdiskDownload/新建文件夹/CCPD2019/splits/train.txt'
    val_path = 'E:/BaiduNetdiskDownload/新建文件夹/CCPD2019/splits/val.txt'
    local_train_path = 'E:/source_common/code/ocr/data/train.txt'
    local_val_path = 'E:/source_common/code/ocr/data/val.txt'
    base_path = 'E:/BaiduNetdiskDownload/新建文件夹/CCPD2019/'
    img_train_path = 'E:/source_common/code/ocr/data/images/train/'
    img_val_path = 'E:/source_common/code/ocr/data/images/val/'
    dataset_split(train_path, local_train_path, img_train_path, base_path)
    print('训练集中各省份车牌占比：', res)
    res = {}
    print('trainset loaded!')
    dataset_split(val_path, local_val_path, img_val_path, base_path)
    print('验证集中各省份车牌占比：', res)
    print('valset loaded!')
