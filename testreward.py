import torch
import cv2
import numpy as np
import pygame

# 從環境檔案中引入環境類別和網路模型類別
from dino import DinoGameEnv
from trainreward import RewardPredictor

class RewardModelTester:
    def __init__(self, model_path, render_mode='human'):
        # 初始化環境
        self.env = DinoGameEnv(render_mode='rgb_array')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 載入預訓練的獎勵預測模型
        self.model = RewardPredictor(action_size=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Error: Model file not found at '{model_path}'")
            quit()
            
        self.model.eval()
        print("Reward Predictor model loaded successfully.")

        # 按鍵映射 (使用 W/S/Space)
        self.key_actions = {
            ord('w'): 1,  # W -> 跳
            ord('s'): 2,  # S -> 蹲
            ord(' '): 0,  # 空白鍵 -> 跑
        }
        self.action_names = {0: "Run", 1: "Jump", 2: "Duck"}

        # 儲存最近兩幀用於預測
        self.s_t = None
        self.s_t_minus_1 = None

    def get_frame_and_preprocess(self):
        """從環境獲取真實畫面並預處理成模型輸入格式"""
        frame_numpy = self.env._get_state()
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        return frame_tensor.to(self.device)

    def run_interactive_test(self):
        print("\n--- Interactive Reward Predictor Test ---")
        print("Control Keys (in the 'Live Game' window):")
        print("  W:         Jump")
        print("  S:         Duck")
        print("  SPACE Bar: Run (Do Nothing)")
        print("  ESC:       Exit")

        # 初始化/重置環境
        self.env.reset()
        self.s_t_minus_1 = self.get_frame_and_preprocess()
        self.s_t = self.get_frame_and_preprocess()

        cv2.namedWindow("Live Game", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Prediction Info", cv2.WINDOW_NORMAL)

        while True:
            # 顯示即時的真實遊戲畫面
            live_frame_raw = np.transpose(pygame.surfarray.array3d(self.env.screen), (1, 0, 2))
            live_frame_bgr = cv2.cvtColor(live_frame_raw, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Game", live_frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC 鍵
                break
            
            action_to_take = None
            # 在每一幀都預測所有可能動作的獎勵
            predicted_rewards = {}
            with torch.no_grad():
                for act in range(self.env.action_space):
                    action_tensor = torch.tensor([[act]], device=self.device, dtype=torch.long)
                    pred_r = self.model(self.s_t_minus_1, self.s_t, action_tensor).item()
                    predicted_rewards[self.action_names[act]] = pred_r

            # 顯示預測面板
            self.display_info_panel(predicted_rewards)

            # 檢查是否有玩家輸入
            if key in self.key_actions:
                action_to_take = self.key_actions[key]
                
            if action_to_take is not None:
                # 玩家選擇了一個動作，實際執行它
                _, true_reward, done = self.env.step(action_to_take)
                
                # 更新畫面歷史
                self.s_t_minus_1 = self.s_t
                self.s_t = self.get_frame_and_preprocess()

                # 打印結果到終端機
                pred_reward_for_action = predicted_rewards[self.action_names[action_to_take]]
                print("-" * 30)
                print(f"Action Taken: {self.action_names[action_to_take]}")
                print(f"  -> True Reward:   {true_reward:.4f}")
                print(f"  -> Predicted Reward: {pred_reward_for_action:.4f}")
                print(f"  -> Prediction Error: {abs(true_reward - pred_reward_for_action):.4f}")

                if done:
                    print("\nGame Over! Resetting environment...")
                    self.env.reset()
                    self.s_t_minus_1 = self.get_frame_and_preprocess()
                    self.s_t = self.get_frame_and_preprocess()
                    print("-" * 30)

        cv2.destroyAllWindows()

    def display_info_panel(self, rewards_dict):
        """創建並顯示一個包含所有預測獎勵的資訊面板"""
        panel = np.full((200, 300, 3), 240, dtype=np.uint8)
        
        y_pos = 30
        for action_name, reward_val in rewards_dict.items():
            text = f"{action_name}: {reward_val:.4f}"
            # 根據獎勵正負給予顏色
            color = (0, 180, 0) if reward_val > 0 else (0, 0, 180)
            if abs(reward_val) < 0.1: # 對微小的生存獎勵用黑色
                 color = (0, 0, 0)
            
            cv2.putText(panel, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 40

        cv2.imshow("Prediction Info", panel)


if __name__ == "__main__":
    MODEL_PATH = "dino_reward_models/best_reward_model.pth"
    
    # 初始化 Pygame
    pygame.init()

    tester = RewardModelTester(model_path=MODEL_PATH)
    tester.run_interactive_test()