import torch.nn as nn
import torch

# In project 5, you need to adjust the model architecture.
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        ################################ you need to modify the cnn model here ################################

        # after convolutoin, the feature map size = ((origin + padding*2 - kernel_size) / stride) + 1
        # input_shape=(3,224,224)
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (3,224,224) -> (32,112,112)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64,112,112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # ResNet + Xception mixed blocks
        self.block1 = nn.Sequential(
            XceptionResidualBlock(64, 128, downsample=True),  # (64,112,112) -> (128,56,56)
        )
        self.block2 = nn.Sequential(
            XceptionResidualBlock(128, 256, downsample=True),  # (128,56,56) -> (256,28,28)
        )
        self.block3 = nn.Sequential(
            XceptionResidualBlock(256, 512, downsample=True),  # (256,28,28) -> (512,14,14)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 12),
        )
        # self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)   # ((224+2*1-3)/1)+1=224  # output_shape=(64,224,224)
        # self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # output_shape=(64,112,112) # (224)/2
        #
        # self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   # output_shape=(128,112,112)
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)    # output_shape=(64,56,56)
        #
        # self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   # output_shape=(64,56,56)
        # self.relu3 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)    # output_shape=(64,28,28)
        #
        # self.fc1 = nn.Linear(64*28*28, 512)
        # self.relu4 = nn.ReLU()
        # self.fc2 = nn.Linear(512, 512)
        # self.relu5 = nn.ReLU()
        # self.fc3 = nn.Linear(512, 12)

        # =================================================================================================== #

    def forward(self, x):
        
        ################################ you need to modify the cnn model here ################################
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
        # out = self.cnn1(x)
        # out = self.relu1(out)
        # out = self.maxpool1(out)
        # out = self.cnn2(out)
        # out = self.relu2(out)
        # out = self.maxpool2(out)
        # out = self.cnn3(out)
        # out = self.relu3(out)
        # out = self.maxpool3(out)
        #
        # out = torch.flatten(out, 1)
        # out = self.fc1(out)
        # out = self.relu4(out)
        # out = self.fc2(out)
        # out = self.relu5(out)
        # out = self.fc3(out)
        # =================================================================================================== #

        return out


class XceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # groups=in_channels：每一個 channel 自己卷自己 → 對應圖中的每個 3x3
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # 1x1 conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class XceptionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample

        stride = 2 if downsample else 1

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # shortcut path (identity mapping)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)

        out += identity
        return self.relu(out)