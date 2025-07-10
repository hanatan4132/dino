import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. 獎勵預測網路架構 ---

class RewardPredictor(nn.Module):
    def __init__(self, action_size=3, embedding_dim=32):
        super(RewardPredictor, self).__init__()
        # 影像編碼器 (可以和影像預測模型共用或獨立)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4), # 輸入 2 幀, [B, 2, 128, 128] -> [B, 32, 31, 31]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> [B, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> [B, 64, 12, 12]
            nn.ReLU(),
            nn.Flatten() # -> [B, 64 * 12 * 12] = [B, 9216]
        )
        
        # 動作嵌入層
        self.action_embedding = nn.Embedding(action_size, embedding_dim)
        
        # 全連接層 (分類器)
        self.fc = nn.Sequential(
            nn.Linear(9216 + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出一個數值，代表預測的 reward
        )

    def forward(self, s_t, s_t_plus_1, action):
        # 將兩幀影像在 channel 維度上合併
        x = torch.cat([s_t, s_t_plus_1], dim=1)
        # 編碼影像
        encoded_image = self.encoder(x)
        
        # 嵌入動作
        # action shape: [B, 1] -> squeeze(1) -> [B]
        action = action.squeeze(1)
        embedded_action = self.action_embedding(action)
        
        # 合併影像特徵和動作特徵
        combined = torch.cat([encoded_image, embedded_action], dim=1)
        
        # 預測獎勵
        predicted_reward = self.fc(combined)
        return predicted_reward

# --- 2. 專為獎勵預測設計的 Dataset 類 ---

class RewardDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # 獎勵預測需要 s_t, s_t+1, a_t -> r_t+1
        self.s_t = data['s_t']
        self.s_t_plus_1 = data['s_t_plus_1']
        self.a_t = data['a_t']
        self.r_t_plus_1 = data['r_t_plus_1']

        print(f"Reward dataset loaded. Number of samples: {len(self.a_t)}")

    def __len__(self):
        return len(self.a_t)

    def __getitem__(self, idx):
        def process_frame(frame):
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame.squeeze(2)
            tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
            return tensor

        s_t = process_frame(self.s_t[idx])
        s_t_plus_1 = process_frame(self.s_t_plus_1[idx])
        
        action = torch.tensor(self.a_t[idx], dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(self.r_t_plus_1[idx], dtype=torch.float32).unsqueeze(0)
        
        return s_t, s_t_plus_1, action, reward

# --- 3. 訓練函式 ---

def train_reward_predictor(model, dataloader, optimizer, criterion, num_epochs, device, save_dir="dino_reward_models", patience=20):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    train_losses = []
    best_loss = float('inf')
    no_improvement_count = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.ion()

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        
        for s_t, s_t_plus_1, action, reward_true in dataloader:
            s_t = s_t.to(device)
            s_t_plus_1 = s_t_plus_1.to(device)
            action = action.to(device)
            reward_true = reward_true.to(device)

            optimizer.zero_grad()
            
            # 模型預測
            pred_reward = model(s_t, s_t_plus_1, action)
            
            # 計算損失
            loss = criterion(pred_reward, reward_true)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")

        # 早停和模型儲存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_reward_model.pth'))
            print(f"New best reward model saved with loss: {best_loss:.6f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

        # 定期視覺化
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 取最後一個 batch 的第一個樣本
                s_t_sample = s_t[0].unsqueeze(0)
                s_t_plus_1_sample = s_t_plus_1[0].unsqueeze(0)
                action_sample = action[0].unsqueeze(0)
                reward_true_sample = reward_true[0].item()
                
                pred_reward_sample = model(s_t_sample, s_t_plus_1_sample, action_sample).item()

                def to_img(tensor):
                    return tensor.cpu().squeeze().numpy()

                # 顯示輸入的兩幀
                axes[0].imshow(to_img(s_t_sample), cmap='gray')
                axes[0].set_title("s_t")
                axes[0].axis('off')
                
                axes[1].imshow(to_img(s_t_plus_1_sample), cmap='gray')
                axes[1].set_title("s_t+1")
                axes[1].axis('off')

                # 顯示預測結果
                action_names = ["Run", "Jump", "Duck"]
                act_name = action_names[action_sample.item()]
                
                axes[2].clear()
                axes[2].set_xlim([0, 1])
                axes[2].set_ylim([0, 1])
                text = (
                    f"Action: {act_name}\n\n"
                    f"True Reward: {reward_true_sample:.2f}\n\n"
                    f"Pred Reward: {pred_reward_sample:.2f}\n\n"
                    f"Error: {abs(reward_true_sample - pred_reward_sample):.2f}"
                )
                axes[2].text(0.1, 0.5, text, fontsize=12, va='center')
                axes[2].set_title("Reward Prediction")
                axes[2].axis('off')
                
                fig.suptitle(f'Epoch {epoch+1} | Loss: {avg_loss:.4f}')
                plt.draw()
                plt.pause(0.01)

    plt.ioff()
    print("Training finished.")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("Reward Predictor Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward_loss_curve.png"))
    plt.show()

# --- 4. 主執行區塊 ---

if __name__ == "__main__":
    # --- 超參數設定 ---
    DATASET_NPZ_PATH = "dino_dataset.npz"
    SAVE_DIR = "dino_reward_models"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 300
    PATIENCE = 30
    
    # --- 初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATASET_NPZ_PATH):
        print(f"Error: Dataset file not found at '{DATASET_NPZ_PATH}'")
    else:
        dataset = RewardDataset(npz_path=DATASET_NPZ_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        model = RewardPredictor(action_size=3).to(device)
        # 對於回歸問題，MSELoss 或 SmoothL1Loss 都是很好的選擇
        # SmoothL1Loss 對於離群值（例如+30或-30的獎勵）更不敏感，更穩健
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_reward_predictor(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=NUM_EPOCHS,
            device=device,
            save_dir=SAVE_DIR,
            patience=PATIENCE
        )