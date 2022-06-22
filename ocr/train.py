import torch
from torch.nn import CTCLoss
from config import common_config, train_config,
from nets.crnn import CRNN
from torch.optim import Adam
from dataset import get_loader
from torch.nn.functional import log_softmax
import os
from evaluate import evaluate


def train_step(net, data, optimizer, criterion, device):
    net.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    outputs = net(images)
    log_probs = log_softmax(outputs, dim=2)
    batch_size = images.size(0)
    input_lengths = torch.LongTensor([outputs.size(0)] * batch_size)
    # target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == '__main__':
    pre_acc = -1.0
    DEVICE = common_config['device']
    print(DEVICE)
    net = CRNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)

    # 载入预训练模型
    if train_config['reload_checkpoint']:
        net.load_state_dict(torch.load(
            train_config['reload_checkpoint'], map_location=DEVICE))
        print('pretrained model loaded over!')

    net.to(DEVICE)
    optimizer = Adam(net.parameters(), lr=train_config['lr'])
    criterion = CTCLoss(blank=0, reduction='sum')
    train_loader, val_loader = get_loader()

    count = 1
    num_epoch = train_config['epochs']

    for epoch in range(num_epoch):
        print(f'begin epoch {epoch} / {num_epoch}')
        total_train_loss = 0
        total_train_size = 0
        for i, data in enumerate(train_loader):
            loss = train_step(net, data, optimizer, criterion, DEVICE)
            batch_size = data[0].size(0)
            total_train_loss += loss
            total_train_size += batch_size

            # 每隔一定次数打印训练结果
            if i % train_config['show_interval'] == 0:
                print('Epoch[{}/{}]\tstep[{}/{}]\tloss:{:.6f}'.format(epoch,
                      num_epoch, i, len(train_loader), loss/batch_size))
            # 每隔一定次数使用验证集进行验证
            if count % train_config['valid_interval'] == 0:
                metircs = evaluate(
                    net, val_loader, criterion, device=DEVICE, decode_method=train_config['decode_method'], beam_size=train_config['beam_size'])
                print(
                    'valid_evaluation: loss={loss}, acc={acc}'.format(**metircs))
                if pre_acc < metircs['acc'] and metircs['acc'] > 0.9:
                    print('more excellent model appears!')
                    pre_acc = metircs['acc']
                    checkpoint_path = os.path.join(
                        train_config['checkpoints_dir'], 'crnn_{}_loss{:.6f}.pth'.format(count, metircs['loss']))
                    torch.save(net.state_dict(), checkpoint_path)
                    print('save model at ', checkpoint_path)
            # 每隔一定次数进行模型保存,此处显示的loss为上一次验证得到的loss
            if count % train_config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    train_config['checkpoints_dir'], 'crnn_{}_loss{:.6f}.pth'.format(count, metircs['loss']))
                torch.save(net.state_dict(), checkpoint_path)
                print('save model at ', checkpoint_path)
            count += 1
        print('-----epoch{} has trained over-----loss:{:.6f}'.format(epoch,
                                                                     total_train_loss/total_train_size))
