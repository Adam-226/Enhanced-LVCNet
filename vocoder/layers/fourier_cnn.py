import torch
import torch.nn as nn
import torch.fft

class FourierCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FourierCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用傅立叶变换
        x_freq = torch.fft.rfft(x, dim=-1)
        x_real = torch.view_as_real(x_freq).permute(0, 2, 1, 3).reshape(x.size(0), -1, x_freq.size(-1))

        # 应用CNN
        x_conv = self.conv(x_real)
        x_conv = self.relu(x_conv)

        # 将频域信号转换回时域
        x_time = torch.fft.irfft(x_conv, n=x.size(-1), dim=-1)

        return x_time