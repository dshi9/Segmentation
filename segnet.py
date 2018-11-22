from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class Segnet(nn.Module):
    def __init__(self, n_classes):
        super(Segnet, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.init_vgg_weights()

        # decode
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv8_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv9_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv10_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, training=True):
        # encoder
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)

        # decoder
        x = self.upsample(x)
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.upsample(x)
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.upsample(x)
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = F.relu(self.conv8_3(x))
        x = self.upsample(x)
        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        x = self.upsample(x)
        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        return x

    def init_vgg_weights(self):
        vgg16 = models.vgg16(pretrained=True)

        assert self.conv1_1.weight.size() == vgg16.features[0].weight.size()
        self.conv1_1.weight.data = vgg16.features[0].weight.data
        assert self.conv1_1.bias.size() == vgg16.features[0].bias.size()
        self.conv1_1.bias.data = vgg16.features[0].bias.data

        assert self.conv1_2.weight.size() == vgg16.features[2].weight.size()
        self.conv1_2.weight.data = vgg16.features[2].weight.data
        assert self.conv1_2.bias.size() == vgg16.features[2].bias.size()
        self.conv1_2.bias.data = vgg16.features[2].bias.data

        assert self.conv2_1.weight.size() == vgg16.features[5].weight.size()
        self.conv2_1.weight.data = vgg16.features[5].weight.data
        assert self.conv2_1.bias.size() == vgg16.features[5].bias.size()
        self.conv2_1.bias.data = vgg16.features[5].bias.data

        assert self.conv2_2.weight.size() == vgg16.features[7].weight.size()
        self.conv2_2.weight.data = vgg16.features[7].weight.data
        assert self.conv2_2.bias.size() == vgg16.features[7].bias.size()
        self.conv2_2.bias.data = vgg16.features[7].bias.data

        assert self.conv3_1.weight.size() == vgg16.features[10].weight.size()
        self.conv3_1.weight.data = vgg16.features[10].weight.data
        assert self.conv3_1.bias.size() == vgg16.features[10].bias.size()
        self.conv3_1.bias.data = vgg16.features[10].bias.data

        assert self.conv3_2.weight.size() == vgg16.features[12].weight.size()
        self.conv3_2.weight.data = vgg16.features[12].weight.data
        assert self.conv3_2.bias.size() == vgg16.features[12].bias.size()
        self.conv3_2.bias.data = vgg16.features[12].bias.data

        assert self.conv3_3.weight.size() == vgg16.features[14].weight.size()
        self.conv3_3.weight.data = vgg16.features[14].weight.data
        assert self.conv3_3.bias.size() == vgg16.features[14].bias.size()
        self.conv3_3.bias.data = vgg16.features[14].bias.data

        assert self.conv4_1.weight.size() == vgg16.features[17].weight.size()
        self.conv4_1.weight.data = vgg16.features[17].weight.data
        assert self.conv4_1.bias.size() == vgg16.features[17].bias.size()
        self.conv4_1.bias.data = vgg16.features[17].bias.data

        assert self.conv4_2.weight.size() == vgg16.features[19].weight.size()
        self.conv4_2.weight.data = vgg16.features[19].weight.data
        assert self.conv4_2.bias.size() == vgg16.features[19].bias.size()
        self.conv4_2.bias.data = vgg16.features[19].bias.data

        assert self.conv4_3.weight.size() == vgg16.features[21].weight.size()
        self.conv4_3.weight.data = vgg16.features[21].weight.data
        assert self.conv4_3.bias.size() == vgg16.features[21].bias.size()
        self.conv4_3.bias.data = vgg16.features[21].bias.data

        assert self.conv5_1.weight.size() == vgg16.features[24].weight.size()
        self.conv5_1.weight.data = vgg16.features[24].weight.data
        assert self.conv5_1.bias.size() == vgg16.features[24].bias.size()
        self.conv5_1.bias.data = vgg16.features[24].bias.data

        assert self.conv5_2.weight.size() == vgg16.features[26].weight.size()
        self.conv5_2.weight.data = vgg16.features[26].weight.data
        assert self.conv5_2.bias.size() == vgg16.features[26].bias.size()
        self.conv5_2.bias.data = vgg16.features[26].bias.data

        assert self.conv5_3.weight.size() == vgg16.features[28].weight.size()
        self.conv5_3.weight.data = vgg16.features[28].weight.data
        assert self.conv5_3.bias.size() == vgg16.features[28].bias.size()
        self.conv5_3.bias.data = vgg16.features[28].bias.data

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x, training=False))
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data[0]

# define the CNN and move the network into GPU
#  Segnet = Segnet(32)
#  Segnet.cuda()
