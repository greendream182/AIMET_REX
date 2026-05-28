import os

import torch
import torch.nn as nn

# 默认从本文件同级目录的 ../data/ 中加载预先计算好的 band 矩阵，
# 这样脚本无论从哪个 cwd 启动都能找到文件
_DEFAULT_MATRIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class BandConverter(nn.Module):
    def __init__(self, band_num=32, freq_bins=257, fs=16000, matrix_dir: str = None):
        super(BandConverter, self).__init__()
        self.band_num = band_num
        self.freq_bins = freq_bins
        matrix_dir = matrix_dir or _DEFAULT_MATRIX_DIR
        to_band_path = os.path.join(matrix_dir, f"to_band_matrix_erb_{band_num}_{freq_bins}.pt")
        inv_to_band_path = os.path.join(matrix_dir, f"inv_to_band_matrix_erb_{band_num}_{freq_bins}.pt")
        self.to_band_matrix = torch.load(to_band_path)
        self.inv_to_band_matrix = torch.load(inv_to_band_path)

    def forward(self, stft_mag, method="magnitude"):
        """
        :param stft_mag: [B C T F] or [B T F]
        :param method: "magnitude" or "energy"
        :return:
        """
        if method == "energy":
            stft_mag = stft_mag ** 2
        band_mag = torch.matmul(stft_mag, self.to_band_matrix.to(stft_mag.device))

        return band_mag

    def inverse_band_mat(self, band_mag, method="magnitude"):
        """
        :param band_mag:
        :param method:
        :return:
        """
        stft_mag = torch.matmul(band_mag, self.inv_to_band_matrix.to(band_mag.device))  #
        if method == "energy":
            stft_mag = stft_mag ** 0.5
        return stft_mag
