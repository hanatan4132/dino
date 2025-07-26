import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from dino import DinoGameEnv
from vae import VAEImagePredictionNet  # 使用 VAE 模型

class WorldModelTester:
    def __init__(self, model_path, render_mode='human'):
        self.env = DinoGameEnv(render_mode='rgb_array') 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = VAEImagePredictionNet(action_size=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Error: Model file not found at '{model_path}'")
            print("Please make sure you have trained the model and the path is correct.")
            

        self.model.eval()
        print("Model loaded successfully.")

        self.key_actions = {
            ord('w'): 1,
            ord('s'): 2,
            ord(' '): 0,
        }
        self.action_names = {0: "Run", 1: "Jump", 2: "Duck"}

        self.prev_frame_t_minus_1 = None
        self.prev_frame_t = None

    def get_real_frame_and_preprocess(self):
        frame_numpy = self.env._get_state()  # shape: (128, 128, 1)
        frame_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
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

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
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
                    action_tensor = torch.tensor([[initial_action]], device=self.device)
                    imagined_actions.append(initial_action)

                    pred_t_plus_1, pred_t_plus_2, *_ = self.model(imagined_frames[-2], imagined_frames[-1], action_tensor)
                    imagined_frames.extend([pred_t_plus_1, pred_t_plus_2])

                    for i in range(num_rollouts - 1):
                        s_prev = imagined_frames[-2]
                        s_curr = imagined_frames[-1]
                        next_action_in_imagination = random.randint(0, 2)
                        imagined_actions.append(next_action_in_imagination)
                        action_tensor = torch.tensor([[next_action_in_imagination]], device=self.device)
                        next_pred_1, next_pred_2, *_ = self.model(s_prev, s_curr, action_tensor)
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

    def plot_imagined_rollout(self, frames, actions):
        num_frames = len(frames)
        np_frames = [f.cpu().squeeze().numpy() for f in frames]
        cols = 6
        rows = (num_frames + cols - 1) // cols
        plt.figure(figsize=(20, 4 * rows))
        plt.suptitle("Model's Imagination Rollout", fontsize=16)

        for i, frame in enumerate(np_frames):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(frame, cmap="gray")
            title = ""
            if i == 0:
                title = "s_t-1 (Real)"
            elif i == 1:
                title = "s_t (Real)"
            else:
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
    MODEL_PATH = "vae_models/best_vae_model.pth"
    import pygame
    pygame.init()

    tester = WorldModelTester(model_path=MODEL_PATH)
    tester.run_interactive_test(num_rollouts=5)