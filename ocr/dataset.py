import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from ocr.config import train_config


class LicensePlate(Dataset):
    # 类的成员变量，可以由类名直接调用
    CHARS = [
        '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁',
        '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
        #'警', '学',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        # len：31 + 25 + 10 = 66
    ]
    # 此处将0号位置空出作为CTC-blank
    CHARS2LABEL = {ch: i + 1 for i, ch in enumerate(CHARS)}
    LABEL2CHAR = {label: ch for ch, label in CHARS2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, transform=None, img_height=32, img_width=100):
        super(LicensePlate, self).__init__()
        if root_dir and mode and not paths:
            paths, texts = self.load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None
        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def load_from_raw_files(self, root_dir, mode):
        # 从train.txt或val.txt中读入路径列表paths和标签列表val.txt
        paths_file = None
        if mode == 'train':
            paths_file = 'train.txt'
        elif mode == 'val':
            paths_file = 'val.txt'
        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r', encoding='utf-8') as f:
            for line in f:
                single_path, single_text = line.strip().split(';')
                paths.append(single_path)
                texts.append(single_text)
        return paths, texts

    def __getitem__(self, index):
        img_path = self.paths[index]
        try:
            image = Image.open(img_path).convert('L')  # 转灰度图
            if self.transform:
                image = self.transform(image)
        except IOError:
            print('load image failed')
        if self.texts:
            text = self.texts[index]
            target = [self.CHARS2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

    def __len__(self):
        return len(self.paths)


train_transform = transforms.Compose(
    [transforms.Resize([32, 100]),
     transforms.ToTensor(),  # 将PIL或者ndarray转换为0~1之间的tensor
     transforms.Normalize((0.5), (0.5))]  # 对每个通道将像素值缩放到-1~1之间
)
val_transform = transforms.Compose(
    [transforms.Resize([32, 100]),
     transforms.ToTensor(),  # 将PIL或者ndarray转换为0~1之间的tensor
     transforms.Normalize((0.5), (0.5))]  # 对每个通道将像素值缩放到-1~1之间
)


def custom_collate_fn(batch: list):
    # 变更Dataloader取出每个batch数据的维度适应CTCLoss输入要求
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    '''
    images：(batch_size , channel , height , width)
    targets：(sum(target_lengths))
    target_lengths：(batch_size)
    '''
    return images, targets, target_lengths


def get_loader():
    root_dir = './data'
    trainset = LicensePlate(root_dir=root_dir, mode='train',
                            transform=train_transform)
    train_loader = DataLoader(dataset=trainset, batch_size=train_config['train_batch_size'], shuffle=True, num_workers=train_config['cpu_workers'],
                              collate_fn=custom_collate_fn)
    valset = LicensePlate(root_dir=root_dir, mode='val',
                          transform=val_transform)
    val_loader = DataLoader(dataset=valset, batch_size=train_config['val_batch_size'], shuffle=False, num_workers=train_config['cpu_workers'],
                            collate_fn=custom_collate_fn)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_loader()
    for i, (images, targets, target_lengths) in enumerate(train_loader):
        break
    print(images.shape)
    print(targets.shape)
    print(target_lengths.shape)

    print(target_lengths)