from torch import nn

#输入图片为1*28*28
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层
        self.convModel = nn.Sequential(
            # 因为simoid容易出现梯度消失问题所以我们使用ReLu
            # N*1*28*28，padding=2后将保持后面的模型不变
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5),padding=2, bias=True),
            nn.ReLU(inplace=True),
            # N*6*28*28
            nn.MaxPool2d(kernel_size=(2, 2)),
            # N*6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), bias=True),
            nn.ReLU(inplace=True),
            # N*16*10*10
            nn.MaxPool2d(kernel_size=(2, 2)),
            # N*16*5*5
        )
        # 全连接层
        self.lineNet = nn.Sequential(
            nn.Flatten(),
            # N*400
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(inplace=True),
            # N*120
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            # N*84
            nn.Linear(in_features=84, out_features=10)
            # N*10
        )
    def forward(self,inputs):
        inputs=self.convModel(inputs)
        outputs=self.lineNet(inputs)
        return outputs

#输入图片为3*32*32（原版的LeNet）
class LeNetOrigin(nn.Module):
    def __init__(self):
        super(LeNetOrigin, self).__init__()
        # 卷积层
        self.convModel = nn.Sequential(
            # 因为simoid容易出现梯度消失问题所以我们使用ReLu
            # N*3*32*32
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), bias=True),
            nn.ReLU(inplace=True),
            # N*6*28*28
            nn.MaxPool2d(kernel_size=(2, 2)),
            # N*6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), bias=True),
            nn.ReLU(inplace=True),
            # N*16*10*10
            nn.MaxPool2d(kernel_size=(2, 2)),
            # N*16*5*5
        )
        # 全连接层
        self.lineNet = nn.Sequential(
            nn.Flatten(),
            # N*400
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(inplace=True),
            # N*120
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            # N*84
            nn.Linear(in_features=84, out_features=10)
            # N*10
        )
        def forward(self, inputs):
            inputs = self.convModel(inputs)
            outputs = self.lineNet(inputs)
            return outputs

