import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from dino import DinoGameEnv
from trainvae import ConditionalVAE
import pygame
class VAETester:
    def __init__(self, model_path, latent_dim):
        self.env = DinoGameEnv(render_mode='human')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用設備: {self.device}")

        self.model = ConditionalVAE(latent_dim=latent_dim, action_dim=1, in_channels=2, out_channels=2).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"錯誤: 在 '{model_path}' 找不到模型檔案。請確認路徑正確。")
            quit()
        self.model.eval()
        print("VAE 世界模型載入成功。")

        self.action_names = {0: "Run", 1: "Jump", 2: "Duck"}
        self.current_input_stack = None

    def _preprocess_frame(self, frame_numpy):
        """將單幀 NumPy 畫面 (128, 128, 1) 預處理成 (1, 1, 128, 128) 的 Tensor"""
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
        return frame_tensor.unsqueeze(0).to(self.device)

    def _get_and_update_stack(self, new_frame_numpy):
        """使用新畫面更新狀態堆疊"""
        new_frame_tensor = self._preprocess_frame(new_frame_numpy)
        if self.current_input_stack is None:
            self.current_input_stack = torch.cat([new_frame_tensor, new_frame_tensor], dim=1)
        else:
            self.current_input_stack = torch.cat([self.current_input_stack[:, 1:2, :, :], new_frame_tensor], dim=1)

    def run_interactive_test(self, num_rollouts=8):
        print("\n--- VAE 世界模型互動測試 ---")
        print("在 Pygame 視窗中操作:")
        print("  ↑ (上方向鍵): 跳躍")
        print("  ↓ (下方向鍵): 蹲下")
        print("  (放開按鍵):   跑步")
        print("  按任意鍵觸發想像，ESC 退出")
        print("-" * 30)
        
        state = self.env.reset()
        self._get_and_update_stack(state)

        while True:
            # 預設動作
            action_to_take = 0
            
            # 使用 Pygame 的事件處理來保持視窗回應
            should_imagine = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    # 按下任何非 ESC 鍵都會觸發想像
                    should_imagine = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action_to_take = 1
            elif keys[pygame.K_DOWN]:
                action_to_take = 2

            if should_imagine:
                print(f"玩家動作: {self.action_names[action_to_take]}。模型開始想像 {num_rollouts} 步...")
                
                imagined_stacks = []
                imagined_rewards = []
                current_imagined_stack = self.current_input_stack.clone()

                with torch.no_grad():
                    # 第一次使用玩家的真實動作
                    action_tensor = torch.tensor([[action_to_take]], dtype=torch.float32, device=self.device)
                    pred_stack, pred_reward, _, _ = self.model(current_imagined_stack, action_tensor)
                    imagined_stacks.append(pred_stack)
                    imagined_rewards.append(pred_reward.item())
                    current_imagined_stack = pred_stack

                    # 後續的想像使用隨機動作
                    for _ in range(num_rollouts - 1):
                        next_action = random.choice(list(self.action_names.keys()))
                        action_tensor = torch.tensor([[next_action]], dtype=torch.float32, device=self.device)
                        pred_stack, pred_reward, _, _ = self.model(current_imagined_stack, action_tensor)
                        imagined_stacks.append(pred_stack)
                        imagined_rewards.append(pred_reward.item())
                        current_imagined_stack = pred_stack
                        print(pred_reward)
                
                self.plot_imagined_rollout(self.current_input_stack, imagined_stacks, imagined_rewards)
            
            # 在真實環境中執行動作並更新狀態
            next_state, _, done = self.env.step(action_to_take)
            self._get_and_update_stack(next_state)

            if done:
                print("遊戲結束! 重置環境...")
                state = self.env.reset()
                self._get_and_update_stack(state)

    def plot_imagined_rollout(self, real_stack, imagined_stacks, imagined_rewards):
        # 從堆疊中分離出單獨的幀
        real_f1 = real_stack[:, 0:1, :, :]
        real_f2 = real_stack[:, 1:2, :, :]
        
        imagined_frames = []
        for stack in imagined_stacks:
            imagined_frames.append(stack[:, 0:1, :, :]) # t+2, t+4, ...
            imagined_frames.append(stack[:, 1:2, :, :]) # t+3, t+5, ...

        all_frames_tensors = [real_f1, real_f2] + imagined_frames
        all_frames_np = [f.cpu().squeeze().numpy() for f in all_frames_tensors]
        
        num_frames = len(all_frames_np)
        cols = 8
        rows = (num_frames + cols - 1) // cols
        
        plt.figure(figsize=(24, 3.5 * rows))
        plt.suptitle("模型想像序列 (Model's Imagination Rollout)", fontsize=16)
        
        for i, frame in enumerate(all_frames_np):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(frame, cmap="gray")
            
            title = ""
            if i == 0: title = "s_t-1 (真實)"
            elif i == 1: title = "s_t (真實)"
            else:
                reward_idx = (i - 2) // 2
                reward_val = imagined_rewards[reward_idx]
                title = f"預測 s_t+{i}\n預測獎勵: {reward_val:.2f}"

            plt.title(title, fontsize=10)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# --- 3. 主執行區塊 ---
if __name__ == "__main__":
    MODEL_PATH = "checkpoints_vae/best_cvae_model.pth" 
    LATENT_DIM = 64 # **極其重要**: 必須與您訓練時使用的維度相同

    tester = VAETester(model_path=MODEL_PATH, latent_dim=LATENT_DIM)
    tester.run_interactive_test(num_rollouts=12) # 讓模型連續想像 7 次