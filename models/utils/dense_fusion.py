import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseFusion(nn.Module):
    def __init__(self, num_points, out_channel=256):
        super(DenseFusion, self).__init__()
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(256, 64, 1)   # (32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

        self.out_conv1 = torch.nn.Conv1d(1408, 640, 1)
        self.out_conv2 = torch.nn.Conv1d(640, out_channel, 1)
    def forward(self, x, emb):
        # x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))     # 256 -> 64
        pointfeat_1 = torch.cat((x, emb), dim=1)    # 64 + 64 = 128

        x = F.relu(self.conv2(x))           # 64 -> 128
        emb = F.relu(self.e_conv2(emb))     # 64 -> 128
        pointfeat_2 = torch.cat((x, emb), dim=1)    # 128 + 128 = 256

        x = F.relu(self.conv5(pointfeat_2)) # 256 -> 512
        x = F.relu(self.conv6(x))           # 512 -> 1024

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        out = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #  128 + 256 + 1024
        out = F.relu(self.out_conv1(out))
        out = F.relu(self.out_conv2(out))

        return out