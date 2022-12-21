import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, dilation=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, stride=1, dilation=2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, stride=1, dilation=3)

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.conv12 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        Eltwise1 = self.relu(conv1 + conv3)

        conv4 = self.conv4(Eltwise1)
        conv5 = self.conv5(conv4)

        Eltwise2 = self.relu(Eltwise1 + conv5)

        conv6 = self.conv6(Eltwise2)
        conv7 = self.conv7(conv6)

        Eltwise3 = self.relu(Eltwise2 + conv7)
        conv8 = self.conv8(Eltwise3)
        conv9 = self.conv9(conv8)

        Eltwise4 = self.relu(Eltwise3 + conv9)
        conv10 = self.conv10(Eltwise4)
        conv11 = self.conv11(conv10)

        Eltwise5 = self.relu(Eltwise4 + conv11)

        Eltwise6 = Eltwise5 + conv1
        conv12 = self.conv12(Eltwise6)

        x = x + conv12

        return x


if __name__ == '__main__':
    from torchstat import stat
    #
    model = SRCNN().cuda()
    stat(model, [[1, 1, 80, 80]])

    # from UDL.UDL.derain.compared_CNN.demo_JORDER_release.caffemodel2pytorch import Net
    #
    # model = Net(
    #     prototxt='D:/ProjectSets/NDA/Attention/compared_100L/demo_JORDER_release/SRCNN_model/deploy_rain_removal_single.prototxt',
    #     weights='D:/ProjectSets/NDA/Attention/compared_100L/demo_JORDER_release/SRCNN_model/rain_removal_single_light.caffemodel',
    #     caffe_proto='https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
    # )
    # print(model)
    # stat(model, [[1, 1, 80, 80]])