import torch
from torch import nn
from pytorch_model_summary import summary


class BidirectionalLSTM(nn.Module):
    def __init__(self, features, hiddens, outputs):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(features, hiddens, bidirectional=True)
        # 多加一层全连接, 为了适应形状
        self.embedding = nn.Linear(hiddens * 2, outputs)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        L, N, H = recurrent.size()
        x = recurrent.view(L * N, H)
        x = self.embedding(x)
        output = x.view(L, N, -1)
        return output


class CRNN(nn.Module):
    # 输入维度为 C * H * W = (1,32,W)
    def __init__(self, input_c, input_h, num_classes, rnn_hidden=256, leaky_relu=False, num_rnn=2):
        super(CRNN, self).__init__()
        assert input_h % 16 == 0, 'input_h has to be a multiple of 16'
        self.input_c = input_c
        self.input_h = input_h
        self.leaky_relu = leaky_relu
        self.cnn = self.cnn_module()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(
                features=512, hiddens=rnn_hidden, outputs=rnn_hidden),
            BidirectionalLSTM(
                features=rnn_hidden, hiddens=rnn_hidden, outputs=num_classes)
        )

    def cnn_module(self):
        params = {
            'kernel_size': [3, 3, 3, 3, 3, 3, 2],
            'padding': [1, 1, 1, 1, 1, 1, 0],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'channel': [64, 128, 256, 256, 512, 512, 512]
        }
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            input_channel = self.input_c if i == 0 else params['channel'][i - 1]
            output_channel = params['channel'][i]
            cnn.add_module('conv{}'.format(i),
                           nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=params['kernel_size'][i], stride=params['stride'][i], padding=params['padding'][i]))
            if batchNormalization:
                cnn.add_module('batchnorm{}'.format(
                    i), nn.BatchNorm2d(output_channel))
            if self.leaky_relu:
                cnn.add_module('relu{}'.format(
                    i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(
            kernel_size=2, stride=2))     # 64, h // 2, w // 2
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2)
                       )     # 128, h // 4, w // 4
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(
            2, 2), stride=(2, 1), padding=(0, 1)))     # 256, h // 8, w // 4
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3', nn.MaxPool2d(
            (2, 2), (2, 1), (0, 1)))    # 512, h // 16, w // 4
        convRelu(6, True)               # 512, h // 16 - 1, *
        return cnn

    def forward(self, x):
        conv_features = self.cnn(x)
        # B, 512, 1,
        B, C, H, W = conv_features.size()
        assert H == 1, "the height of conv_features must be 1"
        conv_features = conv_features.squeeze(2)
        # 把图像的宽度作为类似文本的长度传入LSTM, 通道数则成为了特征
        conv_features = conv_features.permute(2, 0, 1)    # [W,B,C]
        return self.rnn(conv_features)
        # L * B * num_classes


if __name__ == '__main__':
    net = CRNN(input_c=1, input_h=32, num_classes=1000)

    print(summary(net, torch.zeros(5, 1, 32, 100)))
