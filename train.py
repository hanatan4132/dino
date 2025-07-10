import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. 網路架構佔位符 (之後你需要實現這些網路) ---

class ImagePredictionNet(nn.Module):
    def __init__(self, action_size=3):
        super(ImagePredictionNet, self).__init__()
        # TODO: 在這裡定義你的網路架構
        # 範例：一個簡單的 CNN Encoder-Decoder
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1), # 輸入是2幀, [B, 2, 128, 128] -> [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> [B, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> [B, 256, 8, 8]
        )
        # 動作嵌入
        self.action_embedding = nn.Linear(action_size, 256 * 8 * 8)
        
        # Decoder (預測下一幀)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> [B, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> [B, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # -> [B, 1, 128, 128]
            #nn.Sigmoid() # 確保輸出在 0-1 之間
        )
        
        # Decoder (預測下兩幀) - 可以共享部分權重或完全獨立
        # 為了簡單起見，我們用相同的架構
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            #nn.Sigmoid()
        )

    def forward(self, s_t, s_t_plus_1, action):
        # 將兩幀影像在 channel 維度上合併
        x = torch.cat([s_t, s_t_plus_1], dim=1)
        # 編碼
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # --- 關鍵修改部分 ---
        # 1. 確保 action 是一個 1D 的 batch of actions, [B]
        # 原始 action shape: [B, 1] -> squeeze(1) -> [B]
        action = action.squeeze(1)
        
        # 2. 將動作轉換為 one-hot 編碼, shape: [B, num_classes]
        action_one_hot = nn.functional.one_hot(action.to(torch.int64), num_classes=3).float()
        
        # 3. 嵌入 one-hot 編碼, shape: [B, embedding_size]
        action_embedded = self.action_embedding(action_one_hot)
        
        # 現在 encoded_flat 和 action_embedded 都是 2D 張量
        # encoded_flat.shape: [B, 256 * 8 * 8]
        # action_embedded.shape: [B, 256 * 8 * 8]
        # 可以安全地拼接
        combined = torch.cat([encoded_flat, action_embedded], dim=1)
        combined_reshaped = combined.view(combined.size(0), -1, 8, 8)
        
        # 解碼，預測未來兩幀
        pred_s_t_plus_2 = self.decoder1(combined_reshaped)
        pred_s_t_plus_3 = self.decoder2(combined_reshaped)
        
        return pred_s_t_plus_2, pred_s_t_plus_3

class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        # TODO: 之後你可以實現獎勵預測網路
        pass
    
    def forward(self, x):
        # TODO: 實現前向傳播
        pass


# --- 2. 專為恐龍遊戲設計的 Dataset 類 ---

class DinoDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # 讀取所有需要的資料
        self.s_t = data['s_t']
        self.s_t_plus_1 = data['s_t_plus_1']
        self.a_t = data['a_t']
        self.s_t_plus_2 = data['s_t_plus_2']
        self.s_t_plus_3 = data['s_t_plus_3']
        # self.r_t_plus_1 = data['r_t_plus_1'] # 影像預測模型暫時用不到

        print(f"Dataset loaded. Number of samples: {len(self.a_t)}")

    def __len__(self):
        return len(self.a_t)

    def __getitem__(self, idx):
        # 將 NumPy 陣列轉換為 PyTorch 張量，並進行必要的格式調整
        
        # 圖像需要從 (H, W, C) 轉為 (C, H, W)，並正規化到 [0, 1]
        def process_frame(frame):
            # 確保 frame 只有 HxW
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame.squeeze(2)
            # 添加 Channel 維度，並轉為 float32
            tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
            return tensor

        s_t = process_frame(self.s_t[idx])
        s_t_plus_1 = process_frame(self.s_t_plus_1[idx])
        s_t_plus_2 = process_frame(self.s_t_plus_2[idx])
        s_t_plus_3 = process_frame(self.s_t_plus_3[idx])

        # 動作和獎勵轉換為張量
        action = torch.tensor(self.a_t[idx], dtype=torch.long).unsqueeze(0)
        
        return s_t, s_t_plus_1, action, s_t_plus_2, s_t_plus_3


# --- 3. 修改後的訓練函式 ---

def train_image_predictor(model, dataloader, optimizer, criterion, num_epochs, device, save_dir="dino_world_models", patience=20):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    train_losses = []
    best_loss = float('inf')
    no_improvement_count = 0
    
    # 準備繪圖
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle("Training Progress")
    plt.ion()

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for s_t, s_t_plus_1, action, s_t_plus_2_true, s_t_plus_3_true in dataloader:
            # 將資料移到指定設備
            s_t = s_t.to(device)
            s_t_plus_1 = s_t_plus_1.to(device)
            action = action.to(device)
            s_t_plus_2_true = s_t_plus_2_true.to(device)
            s_t_plus_3_true = s_t_plus_3_true.to(device)

            optimizer.zero_grad()
            
            # 模型預測
            pred_s_t_plus_2, pred_s_t_plus_3 = model(s_t, s_t_plus_1, action)
            
            # 計算損失
            loss = criterion(pred_s_t_plus_2, s_t_plus_2_true) + criterion(pred_s_t_plus_3, s_t_plus_3_true)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")

        # 早停和模型儲存邏輯
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_image_model.pth'))
            print(f"New best model saved with loss: {best_loss:.6f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
        
        # 定期視覺化預測結果 (例如每 5 個 epoch)
        if (epoch ) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 從最後一個 batch 取一個樣本來顯示
                s_t_sample = s_t[0].unsqueeze(0)
                s_t_plus_1_sample = s_t_plus_1[0].unsqueeze(0)
                action_sample = action[0].unsqueeze(0)
                s_t_plus_2_true_sample = s_t_plus_2_true[0].unsqueeze(0)
                s_t_plus_3_true_sample = s_t_plus_3_true[0].unsqueeze(0)
                print(s_t_sample.shape)
                print(s_t_plus_1_sample.shape)
                print(action_sample.shape)
                pred_s_t_plus_2, pred_s_t_plus_3 = model(s_t_sample, s_t_plus_1_sample, action_sample)

                def to_img(tensor):
                    # 將 (1, 1, H, W) 的 tensor 轉為 (H, W) 的 numpy array
                    img = tensor.cpu().squeeze().numpy()
                    return (img * 255).astype(np.uint8)

                imgs = [
                    (to_img(s_t_sample), 's_t'),
                    (to_img(s_t_plus_1_sample), 's_t+1'),
                    (to_img(pred_s_t_plus_2), 'pred s_t+2'),
                    (to_img(s_t_plus_2_true_sample), 'true s_t+2'),
                    (to_img(pred_s_t_plus_3), 'pred s_t+3'),
                    (to_img(s_t_plus_3_true_sample), 'true s_t+3')
                ]

                action_names = ["Run", "Jump", "Duck"]
                act_name = action_names[action_sample.item()]
                fig.suptitle(f"Epoch {epoch+1} - Action: {act_name} - Loss: {avg_loss:.4f}")

                for i, (img, title) in enumerate(imgs):
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(title)
                    axes[i].axis('off')
                
                plt.draw()
                plt.pause(0.01)

    # 訓練結束
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_image_model.pth'))
    plt.ioff()
    print("Training finished.")
    
    # 繪製最終的 Loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()


# --- 4. 主執行區塊 ---

if __name__ == "__main__":
    # --- 超參數設定 ---
    DATASET_NPZ_PATH = "dino_dataset.npz"
    SAVE_DIR = "dino_world_models"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 500
    PATIENCE = 50
    
    # --- 初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 檢查資料集是否存在
    if not os.path.exists(DATASET_NPZ_PATH):
        print(f"Error: Dataset file not found at '{DATASET_NPZ_PATH}'")
        print("Please run 'collect_data.py' first.")
    else:
        # 建立 Dataset 和 DataLoader
        dataset = DinoDataset(npz_path=DATASET_NPZ_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # 建立模型、損失函數和優化器
        model = ImagePredictionNet(action_size=3).to(device)
        criterion = nn.MSELoss() # 均方誤差損失，適合圖像生成
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 開始訓練
        train_image_predictor(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=NUM_EPOCHS,
            device=device,
            save_dir=SAVE_DIR,
            patience=PATIENCE
        )