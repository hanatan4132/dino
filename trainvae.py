import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import torchvision.utils as vutils

# --- 1. 修改後的 ConditionalVAE 模型定義 ---
# 輸入和輸出通道數已更新為 2
class ConditionalVAE(nn.Module):
    """
    修改後的 CVAE。
    - 輸入: 一個包含兩張連續畫面的堆疊張量 (N, 2, 128, 128) 和一個動作 (N, 1)。
    - 輸出: 一個包含預測的下兩張畫面的堆疊張量 (N, 2, 128, 128)、一個獎勵和潛在變數。
    """
    def __init__(self, latent_dim=32, action_dim=1, in_channels=2, out_channels=2):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # --- 編碼器 (Encoder) ---
        # <<< 修改：in_channels 從 1 改為 2 >>>
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> (N, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (N, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> (N, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (N, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_log_var = nn.Linear(512 * 16 *16, latent_dim)

        # --- 解碼器 (Decoder) ---
        self.decoder_fc = nn.Linear(latent_dim + action_dim, 512 * 16 * 16)
        
        # <<< 修改：最後一層的 out_channels 從 1 改為 2 >>>
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1), # 輸出兩張圖
            nn.Sigmoid()
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, action):
        z_action = torch.cat([z, action], dim=1)
        h_decoder = self.decoder_fc(z_action)
        h_decoder = h_decoder.view(h_decoder.size(0), 512, 16,16)
        predicted_screen_stack = self.decoder_deconv(h_decoder)
        predicted_reward = self.reward_predictor(z_action)
        return predicted_screen_stack, predicted_reward

    def forward(self, screen_stack, action):
        mu, log_var = self.encode(screen_stack)
        z = self.reparameterize(mu, log_var)
        predicted_screen_stack, predicted_reward = self.decode(z, action)
        return predicted_screen_stack, predicted_reward, mu, log_var


# --- 2. 整合前處理和新資料邏輯的 Dataset ---
class GameDataset(Dataset):
    """
    自定義 Dataset，載入資料並進行前處理。
    - 輸入 (X): s_t 和 s_t_plus_1 堆疊成的 (2, 128, 128) 張量。
    - 輸出 (Y): s_t_plus_2 和 s_t_plus_3 堆疊成的 (2, 128, 128) 張量。
    - 條件: a_t
    - 預測目標: r_t_plus_1
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.s_t = data['s_t']
        self.s_t_plus_1 = data['s_t_plus_1']
        self.s_t_plus_2 = data['s_t_plus_2']
        self.s_t_plus_3 = data['s_t_plus_3']
        self.a_t = data['a_t']
        self.r_t_plus_1 = data['r_t_plus_1']
        print(f"Dataset loaded. Number of samples: {len(self.a_t)}")

    def __len__(self):
        return len(self.a_t)

    def process_frame(self, frame):
        """
        整合您提供的前處理邏輯。
        """
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame.squeeze(2) # 確保 frame 為 HxW
        # 轉換為 (1, H, W) 的 FloatTensor 並正規化
        tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
        return tensor

    def __getitem__(self, idx):
        # 處理輸入畫面: s_t 和 s_t_plus_1
        s_t_proc = self.process_frame(self.s_t[idx])
        s_t_plus_1_proc = self.process_frame(self.s_t_plus_1[idx])
        # 沿著 channel 維度 (dim=0) 堆疊
        input_stack = torch.cat([s_t_proc, s_t_plus_1_proc], dim=0)

        # 處理目標畫面: s_t_plus_2 和 s_t_plus_3
        s_t_plus_2_proc = self.process_frame(self.s_t_plus_2[idx])
        s_t_plus_3_proc = self.process_frame(self.s_t_plus_3[idx])
        target_stack = torch.cat([s_t_plus_2_proc, s_t_plus_3_proc], dim=0)
        
        # 處理動作和獎勵
        action = torch.tensor([self.a_t[idx]], dtype=torch.float32)
        reward = torch.tensor([self.r_t_plus_1[idx]], dtype=torch.float32)

        return input_stack, action, reward, target_stack


# --- 3. 損失函數 (無需修改) ---
def vae_loss_function(pred_screen, true_screen, pred_reward, true_reward, mu, log_var, beta=1.0):
    screen_loss = F.binary_cross_entropy(pred_screen, true_screen, reduction='sum')
    reward_loss = F.mse_loss(pred_reward, true_reward, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = screen_loss + reward_loss + beta * kld_loss
    return total_loss, screen_loss, reward_loss, kld_loss


# --- 新增：可視化函式 ---
def save_reconstruction_comparison(model, device, vis_batch, epoch, save_dir):
    """使用固定的 vis_batch 來產生和儲存預測結果比較圖"""
    model.eval() # 切換到評估模式
    with torch.no_grad():
        input_frames, action, _, target_frames = vis_batch
        input_frames, action = input_frames.to(device), action.to(device)

        # 取得模型預測
        pred_frames, _, _, _ = model(input_frames, action)
        
        # 將 Tensor 移回 CPU 以便儲存
        target_frames = target_frames.cpu()
        pred_frames = pred_frames.cpu()

        # 我們要比較 target_frames 和 pred_frames
        # 兩個張量的維度都是 (N, 2, 128, 128)
        # 我們將它們交錯排列以便比較
        # 比較 t+2 和 t+3 兩幀
        comparison = torch.cat([
            target_frames[:, 0:1, :, :], # 真實的 t+2
            pred_frames[:, 0:1, :, :],   # 預測的 t+2
            target_frames[:, 1:2, :, :], # 真實的 t+3
            pred_frames[:, 1:2, :, :]    # 預測的 t+3
        ])
        
        # 儲存為網格圖片
        save_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch+1}.png")
        vutils.save_image(comparison, save_path, nrow=8) # nrow 控制每行顯示的圖片數量

# --- 修改後的 train 函式 ---
def train(model, train_loader, val_loader, optimizer, device, epochs, beta, 
          save_interval, model_save_dir, best_model_path, 
          vis_batch, vis_dir):
    
    best_val_loss = float('inf') # 初始化最佳損失為無限大

    for epoch in range(epochs):
        # --- 訓練階段 ---
        model.train() # 將模型設為訓練模式
        train_loss_sum = 0
        for batch_idx, (input_frames, action, reward, target_frames) in enumerate(train_loader):
            # ... (與之前相同的訓練邏輯) ...
            input_frames, action, reward, target_frames = (d.to(device) for d in [input_frames, action, reward, target_frames])
            optimizer.zero_grad()
            pred_frames, pred_reward, mu, log_var = model(input_frames, action)
            loss, _, _, _ = vae_loss_function(pred_frames, target_frames, pred_reward, reward, mu, log_var, beta)
            batch_loss = loss / len(input_frames)
            batch_loss.backward()
            optimizer.step()
            train_loss_sum += batch_loss.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)

        # --- 驗證階段 ---
        model.eval() # 將模型設為評估模式
        val_loss_sum = 0
        with torch.no_grad(): # 在評估時不計算梯度
            for input_frames, action, reward, target_frames in val_loader:
                input_frames, action, reward, target_frames = (d.to(device) for d in [input_frames, action, reward, target_frames])
                pred_frames, pred_reward, mu, log_var = model(input_frames, action)
                loss, _, _, _ = vae_loss_function(pred_frames, target_frames, pred_reward, reward, mu, log_var, beta)
                val_loss_sum += (loss / len(input_frames)).item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- 儲存與可視化 ---
        # 1. 儲存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with Val Loss: {avg_val_loss:.4f}")

        # 2. 定期儲存模型
        if (epoch + 1) % save_interval == 0:
            periodic_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), periodic_path)
            print(f"  -> Periodic checkpoint saved to {periodic_path}")

        # 3. 產生並儲存可視化結果
        save_reconstruction_comparison(model, device, vis_batch, epoch, vis_dir)
        print(f"  -> Reconstruction image saved for epoch {epoch+1}")
        print("-" * 50)

    print("Training finished.")

# --- 主執行區塊 ---
if __name__ == '__main__':
    # --- 超參數與路徑設定 ---
    DATA_PATH = "dino_dataset.npz"
    MODEL_SAVE_DIR = "checkpoints_vae"         # 存放定期儲存模型的資料夾
    BEST_MODEL_PATH = "checkpoints_vae/best_cvae_model.pth" # 最佳模型的完整路徑
    VISUALIZATION_DIR = "visualizations_vae"   # 存放比較圖的資料夾
    
    LATENT_DIM = 64
    ACTION_DIM = 1
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 500
    BETA = 0.1
    SAVE_INTERVAL = 50 # 每 10 個 epoch 儲存一次
    VAL_SPLIT = 0.1   # 20% 的資料作為驗證集

    # --- 建立輸出資料夾 ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # --- 環境設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        print(f"錯誤: 在 '{DATA_PATH}' 找不到資料檔案。")
    else:
        # 1. 建立完整資料集
        full_dataset = GameDataset(npz_path=DATA_PATH)
        
        # 2. 切分訓練/驗證集
        val_size = int(len(full_dataset) * VAL_SPLIT)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"Data split: {train_size} training samples, {val_size} validation samples.")

        # 3. 建立 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # 4. 準備一個固定的批次用於可視化
        # 我們從驗證集取一個批次，這樣每次比較的基準都相同
        vis_batch = next(iter(val_loader))

        # 5. 實例化模型和優化器
        model = ConditionalVAE(latent_dim=LATENT_DIM, in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 6. 開始訓練
        train(model, train_loader, val_loader, optimizer, device, EPOCHS, BETA,
              SAVE_INTERVAL, MODEL_SAVE_DIR, BEST_MODEL_PATH,
              vis_batch, VISUALIZATION_DIR)