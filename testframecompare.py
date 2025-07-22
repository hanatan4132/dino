import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame

# 從環境檔案中引入環境類別和網路模型類別
from dino import DinoGameEnv
from train import ImagePredictionNet

# --- 設定 ---
SEQUENCE_LENGTH = 11 # 定義要連續預測和比較的步數

class WorldModelDeepComparer:
    """
    修改後的測試類，用於進行連續多步預測，並與真實環境的演變進行比較。
    """
    def __init__(self, model_path):
        self.env = DinoGameEnv(render_mode='rgb_array') 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 載入預測模型
        self.model = ImagePredictionNet(action_size=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"錯誤: 在 '{model_path}' 找不到模型檔案")
            print("請確認您已完成模型訓練，且路徑正確。")
            quit()
            
        self.model.eval()
        print("模型成功載入。")

        self.action_names = {0: "Run", 1: "Jump", 2: "Duck"}

    def preprocess_frame(self, frame_numpy):
        """
        將單一 NumPy 畫面進行預處理，轉換為模型所需的 Tensor 格式。
        """
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        return frame_tensor.to(self.device)

    def run_deep_comparison_test(self):
        """
        主執行函式，觸發一個動作序列，並比較真實與預測的結果。
        """
        print("\n--- World Model Deep Prediction vs. Real Environment ---")
        print(f"每次按鍵將執行 {SEQUENCE_LENGTH} 次相同的動作進行比較。")
        print("控制按鍵 (在 'Live Game' 視窗中):")
        print("  W:               執行跳躍序列")
        print("  S:               執行蹲下序列")
        print("  空白鍵:          執行跑步序列")
        print("  ESC:             退出")

        # --- 初始化環境與狀態 ---
        frame_t_minus_1_np = self.env.reset()
        frame_t_np, _, _ = self.env.step(0)
        s_t_minus_1 = self.preprocess_frame(frame_t_minus_1_np)
        s_t = self.preprocess_frame(frame_t_np)

        cv2.namedWindow("Live Game", cv2.WINDOW_NORMAL)

        while True:
            # 顯示即時遊戲畫面
            live_frame_raw = np.transpose(pygame.surfarray.array3d(self.env.screen), (1, 0, 2))
            live_frame_bgr = cv2.cvtColor(live_frame_raw, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Game", live_frame_bgr)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC 鍵
                print("測試結束。")
                break

            initial_action = None
            if key == ord('w'): initial_action = 1
            elif key == ord('s'): initial_action = 2
            elif key == ord(' '): initial_action = 0
            
            if initial_action is not None:
                action_sequence = [initial_action] * SEQUENCE_LENGTH
                print(f"\n玩家選擇動作 '{self.action_names[initial_action]}'. 開始 {SEQUENCE_LENGTH} 步的比較...")

                # --- 1. 執行模型的深度想像 ---
                imagined_frames = []
                # 使用當前的真實狀態作為想像的起點
                imagined_s_t_minus_1 = s_t_minus_1
                imagined_s_t = s_t
                with torch.no_grad():
                    for i in range(SEQUENCE_LENGTH):
                        action_tensor = torch.tensor([[action_sequence[i]]], device=self.device, dtype=torch.long)
                        pred_s_t1, pred_s_t2 = self.model(imagined_s_t_minus_1, imagined_s_t, action_tensor)
                        imagined_frames.extend([pred_s_t1, pred_s_t2])
                        
                        # 將預測的結果作為下一步想像的輸入
                        imagined_s_t_minus_1, imagined_s_t = pred_s_t1, pred_s_t2

                # --- 2. 在真實環境中執行同樣的動作序列 ---
                real_frames = []
                game_over = False
                for action in action_sequence:
                    if game_over: # 如果遊戲中途結束，用最後一幀填充剩餘步驟
                        real_frames.extend([real_frames[-2], real_frames[-1]])
                        continue

                    real_frame_t1_np, _, done1 = self.env.step(action)
                    real_frame_t2_np, _, done2 = (self.env.step(action) if not done1 else (real_frame_t1_np, 0, True))
                    
                    real_frames.append(self.preprocess_frame(real_frame_t1_np))
                    real_frames.append(self.preprocess_frame(real_frame_t2_np))
                    
                    if done1 or done2:
                        game_over = True

                # --- 3. 繪製深度比較圖 ---
                self.plot_deep_comparison(s_t, action_sequence, real_frames, imagined_frames)

                # --- 4. 更新主狀態 ---
                if game_over:
                    print("遊戲結束！正在重置環境...")
                    frame_t_minus_1_np = self.env.reset()
                    frame_t_np, _, _ = self.env.step(0)
                    s_t_minus_1 = self.preprocess_frame(frame_t_minus_1_np)
                    s_t = self.preprocess_frame(frame_t_np)
                else:
                    # 使用真實序列的最後兩幀來更新主狀態
                    s_t_minus_1 = real_frames[-2]
                    s_t = real_frames[-1]

        cv2.destroyAllWindows()
        self.env.close()

    def plot_deep_comparison(self, initial_s_t, actions, real_frames, predicted_frames):
        """將多步真實畫面與預測畫面進行並排比較。"""
        
        def to_numpy(tensor):
            return tensor.cpu().squeeze().numpy()

        plt.figure(figsize=(15, 3 * (SEQUENCE_LENGTH + 1)))
        action_name = self.action_names[actions[0]]
        plt.suptitle(f"Deep Comparison for '{action_name}' Sequence ({SEQUENCE_LENGTH} steps)", fontsize=18)

        # 繪製初始狀態
        plt.subplot(SEQUENCE_LENGTH + 1, 3, 2)
        plt.imshow(to_numpy(initial_s_t), cmap="gray", vmin=0, vmax=1)
        plt.title("Initial State (s_t)", fontsize=14)
        plt.axis('off')

        for i in range(SEQUENCE_LENGTH):
            # 每個 step 包含兩幀 (t+1, t+2)
            real_t1 = real_frames[i * 2]
            real_t2 = real_frames[i * 2 + 1]
            pred_t1 = predicted_frames[i * 2]
            pred_t2 = predicted_frames[i * 2 + 1]

            # --- 左側標題 ---
            ax_title = plt.subplot(SEQUENCE_LENGTH + 1, 3, (i + 1) * 3 + 1)
            ax_title.text(0.5, 0.5, f"Step {i+1}\nAction: {self.action_names[actions[i]]}", 
                          ha='center', va='center', fontsize=14, weight='bold')
            ax_title.axis('off')

            # --- 真實畫面 ---
            ax_real = plt.subplot(SEQUENCE_LENGTH + 1, 3, (i + 1) * 3 + 2)
            # 將兩幀並排顯示在一張圖中
            combined_real = np.concatenate([to_numpy(real_t1), to_numpy(real_t2)], axis=1)
            ax_real.imshow(combined_real, cmap="gray", vmin=0, vmax=1)
            if i == 0: ax_real.set_title("REAL Evolution (s_t+1, s_t+2)", fontsize=14)
            ax_real.axis('off')

            # --- 預測畫面 ---
            ax_pred = plt.subplot(SEQUENCE_LENGTH + 1, 3, (i + 1) * 3 + 3)
            # 將兩幀並排顯示在一張圖中
            combined_pred = np.concatenate([to_numpy(pred_t1), to_numpy(pred_t2)], axis=1)
            ax_pred.imshow(combined_pred, cmap="gray", vmin=0, vmax=1)
            if i == 0: ax_pred.set_title("PREDICTED Evolution (s_t+1, s_t+2)", fontsize=14)
            ax_pred.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    MODEL_PATH = "dino_world_models/best_image_model.pth"
    
    pygame.init()

    tester = WorldModelDeepComparer(model_path=MODEL_PATH)
    tester.run_deep_comparison_test()
    
    pygame.quit()