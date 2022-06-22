'''
    由ccpd_base生成VOC格式.xml文件
'''
import os
from xml.dom.minidom import Document
from tqdm import tqdm


def analysis_file(filename):
    # 解析图片文件名提取车牌的左上角和右小角坐标
    seven_parts = filename.split('-')
    str_coordinates = seven_parts[3]

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
    return coordinates, left_top, right_bottom


def generate(data_path='VOCdevkit/VOC2007/JPEGImages'):
    dirs = os.listdir(data_path)
    bar = tqdm(dirs)
    for i, img_name in enumerate(bar):
        id = img_name[:-4]
        xml_path = 'VOCdevkit/VOC2007/Annotations/' + id + '.xml'
        _, left_top, right_bottom = analysis_file(img_name)
        doc = Document()

        root = doc.createElement('annotation')
        doc.appendChild(root)

        filename_node = doc.createElement('filename')
        filename = doc.createTextNode(img_name)
        filename_node.appendChild(filename)
        root.appendChild(filename_node)

        object_node = doc.createElement('object')
        root.appendChild(object_node)

        name_node = doc.createElement('name')
        name = doc.createTextNode('license')
        name_node.appendChild(name)
        object_node.appendChild(name_node)
        
        bndbox_node = doc.createElement('bndbox')
        object_node.appendChild(bndbox_node)

        xmin_node = doc.createElement('xmin')
        xmin = doc.createTextNode(str(left_top[0]))
        xmin_node.appendChild(xmin)
        bndbox_node.appendChild(xmin_node)

        ymin_node = doc.createElement('ymin')
        ymin = doc.createTextNode(str(left_top[1]))
        ymin_node.appendChild(ymin)
        bndbox_node.appendChild(ymin_node)

        xmax_node = doc.createElement('xmax')
        xmax = doc.createTextNode(str(right_bottom[0]))
        xmax_node.appendChild(xmax)
        bndbox_node.appendChild(xmax_node)

        ymax_node = doc.createElement('ymax')
        ymax = doc.createTextNode(str(right_bottom[1]))
        ymax_node.appendChild(ymax)
        bndbox_node.appendChild(ymax_node)

        with open(xml_path, "w", encoding="utf-8") as f:
            doc.writexml(f, indent='', addindent='\t',
                         newl='\n', encoding="utf-8")
            f.close()
        # bar.set_description(f'{i} / {len(dirs)}')

if __name__ == '__main__':
    generate()