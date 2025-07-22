import torch
import torch.nn as nn
import numpy as np

class DQN_Network(nn.Module):
    """
    一個加深版的卷積神經網路，專為 DQN 設計。
    """
    def __init__(self, input_shape, num_actions):
        """
        初始化網路。
        Args:
            input_shape (tuple): 輸入狀態的形狀，格式為 (C, H, W)。
                                 例如 (4, 128, 128) 代表4幀堆疊的128x128圖像。
            num_actions (int):   可執行的動作數量。
        """
        super(DQN_Network, self).__init__()
        
        self.input_channels = input_shape[0]
        
        # 卷積層 (特徵提取器)
        self.conv_layers = nn.Sequential(
            # Layer 1: 大感受野，快速降低解析度
            # Input: (B, 4, 128, 128) -> Output: (B, 32, 31, 31)
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Layer 2: 增加通道數，進一步降低解析度
            # Input: (B, 32, 31, 31) -> Output: (B, 64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 保持解析度，加深網路
            # Input: (B, 64, 14, 14) -> Output: (B, 64, 12, 12)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Layer 4: 再加一層，提取更抽象的特徵
            # Input: (B, 64, 12, 12) -> Output: (B, 128, 10, 10)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        
        # 使用輔助函式自動計算卷積層輸出的扁平化大小
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # 全連接層 (Q值估計器)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        print("--- Deep CNN DQN Model Initialized ---")
        print(f"Convolutional output size: {conv_output_size}")
        # print(f"Network architecture:\n{self}") # 可以取消註解來查看完整結構
    
    def _get_conv_output_size(self, shape):
        # 創建一個假的張量來通過卷積層，以確定輸出大小
        o = self.conv_layers(torch.zeros(1, *shape))
        # 返回扁平化後的大小
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        conv_out = self.conv_layers(x)
        # 展平，準備送入全連接層
        flattened = conv_out.view(x.size(0), -1)
        # 輸出Q值
        q_values = self.fc_layers(flattened)
        return q_values