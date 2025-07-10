# ... (import 和設定部分保持不變) ...
import numpy as np
import os
import random
from collections import deque
from dino import DinoGameEnv
import pygame
# --- 設定 ---
DATASET_PATH = "dino_dataset.npz"
NUM_EPISODES_TO_COLLECT = 300
MAX_FRAMES_PER_EPISODE = 2000
# <<< 新增設定：模式切換機率 >>>
# 每幀有 1% 的機率切換模式
MODE_SWITCH_PROBABILITY = 0.005

# --- 修改後的 DataCollector 類 ---
class DataCollector:
    def __init__(self, env):
        self.env = env
        # <<< 修改：不再需要固定的 mode 參數 >>>
        self.current_mode = 'expert' # 預設以專家模式開始
        self.dataset = {
            's_t': [],
            's_t_plus_1': [],
            'a_t': [],
            's_t_plus_2': [],
            's_t_plus_3': [],
            'r_t_plus_1': []
        }
        print(f"Data Collector initialized. Mode will be switched dynamically.")

    def expert_policy(self):
        """
        一個簡單的基於規則的專家 AI
        (此函式內容不變)
        """
        player_pos = self.env.player.dino_rect
        scan_box_x = player_pos.right + 10
        scan_box_y_cactus = player_pos.top
        scan_box_y_bird = player_pos.top - 50
        scan_width = 150 + int(self.env.game_speed * 2)
        scan_height = player_pos.height

        for obstacle in self.env.obstacles:
            scan_box_cactus = (scan_box_x, scan_box_y_cactus, scan_width, scan_height)
            scan_box_bird = (scan_box_x, scan_box_y_bird, scan_width, scan_height)
            
            if 'Cactus' in obstacle.__class__.__name__:
                if obstacle.rect.colliderect(scan_box_cactus):
                    return 1
            elif 'Bird' in obstacle.__class__.__name__:
                 if obstacle.rect.colliderect(scan_box_bird):
                    return 2
                        
        return 0

    def choose_action(self):
        """根據當前模式選擇動作"""
        # <<< 修改：根據 self.current_mode 來選擇 >>>
        if self.current_mode == 'expert':
            return self.expert_policy()
        else: # 'random'
            return random.randint(0, self.env.action_space - 1)

    def switch_mode(self):
        """根據機率隨機切換模式"""
        if random.random() < MODE_SWITCH_PROBABILITY:
            if self.current_mode == 'expert':
                self.current_mode = 'random'
                print(f"Mode switched to RANDOM at frame {self.frame_count}")
            else:
                self.current_mode = 'expert'
                print(f"Mode switched to EXPERT at frame {self.frame_count}")


    def collect_episode(self, max_frames):
        """玩一局遊戲並收集資料"""
        state = self.env.reset()
        done = False
        self.frame_count = 0
        duck=0
        # <<< 修改：在每局開始時隨機決定初始模式 >>>
        self.current_mode = random.choice(['expert', 'random'])
        print(f"Episode starting with mode: {self.current_mode}")
        
        frame_buffer = deque(maxlen=4)
        frame_buffer.append(state)

        while not done and self.frame_count < max_frames:
            # <<< 新增：在每一步都嘗試切換模式 >>>
            self.switch_mode()
            
            if len(frame_buffer) < 2:
                next_state, _, done = self.env.step(0)
                frame_buffer.append(next_state)
                self.frame_count += 1
                continue
            if duck < 8 and duck>0:
                action = 2

            else:
                action = self.choose_action()
                duck =0
            if action == 2:
               duck+= 1
            state_plus_2, reward, done = self.env.step(action)
            frame_buffer.append(state_plus_2)

            if not done:
                state_plus_3, reward2, done_after = self.env.step(action)
                if done_after: done = True
            else:
                state_plus_3 = state_plus_2
            frame_buffer.append(state_plus_3)
            
            s_t, s_t_plus_1, s_t_plus_2, s_t_plus_3 = list(frame_buffer)
            if reward2 > 28 or reward2 < -30 :
                reward = reward2
            self.dataset['s_t'].append(s_t)
            self.dataset['s_t_plus_1'].append(s_t_plus_1)
            self.dataset['a_t'].append(action)
            self.dataset['s_t_plus_2'].append(s_t_plus_2)
            self.dataset['s_t_plus_3'].append(s_t_plus_3)
            self.dataset['r_t_plus_1'].append(reward)

            self.frame_count += 1
            self.env.render()

        print(f"Episode finished. Total data points collected: {len(self.dataset['a_t'])}")

    def save_data(self, path):
        """將收集到的資料儲存到檔案"""
        if not self.dataset['a_t']:
            print("No data collected. Skipping save.")
            return

        processed_dataset = {}
        for key, value in self.dataset.items():
            if key.startswith('s_'):
                dtype = np.uint8
            elif key.startswith('a_'):
                dtype = np.int16
            elif key.startswith('r_'):
                dtype = np.float32
            else:
                dtype = None 
            
            processed_dataset[key] = np.array(value, dtype=dtype)

        np.savez_compressed(path, **processed_dataset)
        print(f"Dataset saved to {path}. Shape of s_t: {processed_dataset['s_t'].shape}")

# --- 修改後的主執行區塊 ---
if __name__ == '__main__':
    env = DinoGameEnv(render_mode='human')
    
    # <<< 修改：初始化 Collector 時不再傳入 mode >>>
    collector = DataCollector(env)

    for i in range(NUM_EPISODES_TO_COLLECT):
        print(f"\n--- Starting Episode {i + 1}/{NUM_EPISODES_TO_COLLECT} ---")
        collector.collect_episode(MAX_FRAMES_PER_EPISODE)

    collector.save_data(DATASET_PATH)

    env.render()
    pygame.quit()