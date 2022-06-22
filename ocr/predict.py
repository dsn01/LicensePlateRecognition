from os import path
import torch
import glob
from torch.nn.functional import log_softmax
from torchvision import transforms
from nets.crnn import CRNN
from config import common_config
from dataset import LicensePlate
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
import time
from PIL import Image


pretrained_path = './checkpoint/crnn_25500_loss0.050892.pth'
decode_method = 'beam_search'
beam_size = 10
device = common_config['device']
transform = transforms.Compose(
    [transforms.Resize([32, 100]),
     transforms.ToTensor(),  # 将PIL或者ndarray转换为0~1之间的tensor
     transforms.Normalize((0.5), (0.5))]  # 对每个通道将像素值缩放到-1~1之间
)


def batch_prediction(path):
    print('-----------批量图片识别开始-----------')
    print(device)
    images_path = glob.glob(path)
    batch_size = 32

    demo_dataset = LicensePlate(paths=images_path, transform=transform)
    demo_loader = DataLoader(dataset=demo_dataset,
                             batch_size=batch_size, shuffle=False)
    net = CRNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))
    net.to(device)
    outputs = []
    clock_start = time.time()
    with torch.no_grad():
        for data in demo_loader:
            images = data.to(device)

            logits = net(images)
            log_probs = log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=decode_method,
                               beam_size=beam_size, label2char=LicensePlate.LABEL2CHAR)
            outputs += preds
    print('图片推理平均耗时{:.6f}s'.format((time.time() - clock_start) / len(outputs)))
    i = 0
    for path, pred in zip(images_path, outputs):
        if i > 1000:
            break
        text = ''.join(pred)
        print('{}的预测结果为：{}'.format(path, text))
        i += 1


def single_prediction(path):
    print('-----------单张图片识别开始-----------')
    print(device)
    image = Image.open(path).convert('L')
    image = transform(image).unsqueeze(0)  # 增加batch维度
    # print(type(image),image.shape)
    net = CRNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))
    net.to(device)
    with torch.no_grad():
        image = image.to(device)
        clock_start = time.time()
        logits = net(image)
        print('单张图片推理耗时{}s'.format(time.time()-clock_start))
        log_probs = log_softmax(logits, dim=2)
        pred = ctc_decode(log_probs, method=decode_method,
                          beam_size=beam_size, label2char=LicensePlate.LABEL2CHAR)
        text = ''.join(pred[0])
        print('{}的预测结果为：{}'.format(path, text))


def predict(path, mode='single'):
    '''
    single模式下，path即为图片路径
    batch模式下，path为 './demo/*.jpg'格式
    '''
    if mode == 'single':
        single_prediction(path)
    elif mode == 'batch':
        batch_prediction(path)


if __name__ == '__main__':
    
    path = './31_皖B76840.jpg'
    predict(path, mode='single')
    
    # path = './demo/*.jpg'
    # predict(path, mode='batch')