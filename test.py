import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 從環境檔案中引入環境類別和網路模型類別
from dino import DinoGameEnv
from train import ImagePredictionNet

class WorldModelTester:
    def __init__(self, model_path, render_mode='human'):
        self.env = DinoGameEnv(render_mode='rgb_array') 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ImagePredictionNet(action_size=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Error: Model file not found at '{model_path}'")
            print("Please make sure you have trained the model and the path is correct.")
            quit()
            
        self.model.eval()
        print("Model loaded successfully.")

        self.key_actions = {
            ord('w'): 1,  # W 或 上箭头 -> 跳
            ord('s'): 2,  # S 或 下箭头 -> 蹲
            ord(' '): 0,  # 空白鍵 -> 跑
        }
        # OpenCV 的 waitKey 對方向鍵的編碼比較特殊，我們用 W/S 代替
        self.action_names = {0: "Run", 1: "Jump", 2: "Duck"}

        self.prev_frame_t_minus_1 = None
        self.prev_frame_t = None
    
    def get_real_frame_and_preprocess(self):
        """
        從環境中獲取真實畫面並預處理。
        <<< 這是關鍵的修改部分 >>>
        """
        # 1. 從環境獲取 NumPy 畫面 (H, W, C)
        frame_numpy = self.env._get_state() # shape: (128, 128, 1)
        
        # 2. 轉換為 PyTorch 張量並調整維度 (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
        
        # 3. 增加 Batch 維度 (C, H, W) -> (1, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor.to(self.device)

    def run_interactive_test(self, num_rollouts=5):
        print("\n--- Interactive World Model Test ---")
        print("Control Keys (in the 'Live Game' window):")
        print("  W / UP Arrow:    Jump")
        print("  S / DOWN Arrow:  Duck")
        print("  SPACE Bar:       Run (Do Nothing)")
        print("  ESC:             Exit")

        self.env.reset()
        self.prev_frame_t_minus_1 = self.get_real_frame_and_preprocess()
        self.prev_frame_t = self.get_real_frame_and_preprocess()

        cv2.namedWindow("Live Game", cv2.WINDOW_NORMAL)

        while True:
            live_frame_raw = np.transpose(pygame.surfarray.array3d(self.env.screen), (1, 0, 2))
            live_frame_bgr = cv2.cvtColor(live_frame_raw, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Game", live_frame_bgr)
            
            # 使用 waitKeyEx 來更好地捕捉方向鍵
            key = cv2.waitKey(0) & 0xFF
            
            # 處理方向鍵 (在不同系統上返回值可能不同)
            # 為了簡單和通用，我們主要用 W/S/Space
            if key == 27:  # ESC 鍵
                break

            action_to_take = None
            if key == ord('w'):
                action_to_take = 1
            elif key == ord('s'):
                action_to_take = 2
            elif key == ord(' '):
                action_to_take = 0
            
            if action_to_take is not None:
               initial_action = action_to_take
               print(f"\nPlayer chose action: {self.action_names[initial_action]}. Predicting {num_rollouts} steps...")
               
               imagined_frames = [self.prev_frame_t_minus_1, self.prev_frame_t]
               imagined_actions = []

               with torch.no_grad():
                   # <<< 核心修改 1 >>>
                   # 創建一個 [1, 1] 形狀的 2D 張量
                   action_tensor = torch.tensor([[initial_action]], device=self.device)
                   imagined_actions.append(initial_action)
                   
                   pred_t_plus_1, pred_t_plus_2 = self.model(imagined_frames[-2], imagined_frames[-1], action_tensor)
                   imagined_frames.extend([pred_t_plus_1, pred_t_plus_2])

                   for i in range(num_rollouts - 1):
                       s_prev = imagined_frames[-2]
                       s_curr = imagined_frames[-1]
                       next_action_in_imagination = random.randint(0, 2)
                       imagined_actions.append(next_action_in_imagination)
                       
                       # <<< 核心修改 2 >>>
                       # 同樣創建 [1, 1] 形狀的 2D 張量
                       action_tensor = torch.tensor([[next_action_in_imagination]], device=self.device)

                       next_pred_1, next_pred_2 = self.model(s_prev, s_curr, action_tensor)
                       imagined_frames.extend([next_pred_1, next_pred_2])

               self.plot_imagined_rollout(imagined_frames, imagined_actions)

               _, _, done = self.env.step(initial_action)
               self.prev_frame_t_minus_1 = self.prev_frame_t
               self.prev_frame_t = self.get_real_frame_and_preprocess()

               if done:
                    print("Game Over! Resetting environment.")
                    self.env.reset()
                    self.prev_frame_t_minus_1 = self.get_real_frame_and_preprocess()
                    self.prev_frame_t = self.get_real_frame_and_preprocess()

        cv2.destroyAllWindows()
        self.env.close()

    # (plot_imagined_rollout 方法保持不變)
    def plot_imagined_rollout(self, frames, actions):
        """將想像中的畫面序列繪製出來"""
        num_frames = len(frames)
        # 轉換為 numpy array 以便顯示
        np_frames = [f.cpu().squeeze().numpy() for f in frames]
        
        # 決定 subplot 的行列數
        cols = 6
        rows = (num_frames + cols - 1) // cols
        
        plt.figure(figsize=(20, 4 * rows))
        plt.suptitle("Model's Imagination Rollout", fontsize=16)
        
        for i, frame in enumerate(np_frames):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(frame, cmap="gray", vmin=0, vmax=1)
            
            title = ""
            if i == 0:
                title = "s_t-1 (Real)"
            elif i == 1:
                title = "s_t (Real)"
            else:
                # 標示出是哪個動作導致了這一幀的產生
                action_idx = (i - 2) // 2
                if action_idx < len(actions):
                    act_name = self.action_names[actions[action_idx]]
                    title = f"Pred s_t+{i-1}\n(from A={act_name})"
                else:
                    title = f"Pred s_t+{i-1}"

            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    # 確保模型路徑正確
    MODEL_PATH = "dino_world_models/best_image_model.pth"
    
    # 引入 pygame 以便在主執行緒中初始化
    import pygame
    pygame.init()

    tester = WorldModelTester(model_path=MODEL_PATH)
    tester.run_interactive_test(num_rollouts=5) # 讓模型連續想像5次 (總共會產生 2 + 2*5 = 12 幀)