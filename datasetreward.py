import numpy as np
import os
import random
from collections import deque
from dino import DinoGameEnv  # 假設你的環境在 dino_env.py
import pygame

# --- 設定 ---
# 儲存為新的檔名，以區分資料集
DATASET_PATH = "dino_reward_dataset.npz" 
NUM_EPISODES_TO_COLLECT = 800 # 為了得到足夠的非0.1樣本，可以增加局數
MAX_FRAMES_PER_EPISODE = 2000
MODE_SWITCH_PROBABILITY = 0.005

# <<< 新增設定：降采样（Down-sampling） >>>
REWARD_THRESHOLD = 0.2  # 我們認為絕對值小於這個的獎勵是“低信息量”的
DISCARD_PROBABILITY = 0.8 # 丟棄低信息量樣本的機率 (80%)

class RewardDataCollector:
    def __init__(self, env):
        self.env = env
        self.current_mode = 'expert'
        # 你的專家AI蹲下邏輯需要一個計時器
        self.duck_persistence_frames = 0
        
        # <<< 修改：資料集結構簡化 >>>
        self.dataset = {
            's_t': [],
            's_t_plus_1': [],
            'a_t': [],
            'r_t_plus_1': []
        }
        print("Reward Data Collector initialized with down-sampling.")

    def expert_policy(self):
        # ... (你的 expert_policy 邏輯保持不變) ...
        player_pos = self.env.player.dino_rect
        scan_box_x = player_pos.right + 10
        scan_width = 150 + int(self.env.game_speed * 2)
        scan_height = player_pos.height
        for obstacle in self.env.obstacles:
            scan_box_cactus = (scan_box_x, player_pos.top, scan_width, scan_height)
            scan_box_bird = (scan_box_x, player_pos.top - 50, scan_width, scan_height)
            if 'Cactus' in obstacle.__class__.__name__:
                if obstacle.rect.colliderect(scan_box_cactus): return 1
            elif 'Bird' in obstacle.__class__.__name__:
                if obstacle.rect.colliderect(scan_box_bird): return 2
        return 0

    def choose_action(self):
        # 優先級 1: 強制蹲下
        if self.duck_persistence_frames > 0:
            self.duck_persistence_frames -= 1
            return 2
        
        # 優先級 2: 根據模式決策
        action_to_take = 0
        if self.current_mode == 'expert':
            action_to_take = self.expert_policy()
        else: # random
            action_to_take = random.randint(0, self.env.action_space - 1)
        
        # 如果決策是蹲下，啟動計時器
        if action_to_take == 2:
            self.duck_persistence_frames = 35 # 保持蹲下的幀數
        
        return action_to_take

    def switch_mode(self):
        if random.random() < MODE_SWITCH_PROBABILITY:
            self.current_mode = 'random' if self.current_mode == 'expert' else 'expert'
            # print(f"Mode switched to {self.current_mode.upper()}") # 可以取消註解來除錯

    def collect_episode(self, max_frames):
        state = self.env.reset()
        done = False
        self.frame_count = 0
        self.duck_persistence_frames = 0
        if random.random() < 0.3:
            self.current_mode = 'expert'
        else:
            self.current_mode = 'random'
        
        # 只需要一個能裝兩幀的 buffer
        frame_buffer = deque(maxlen=2)
        frame_buffer.append(state)
        # 執行一步來填滿 buffer
        initial_state, _, _ = self.env.step(0)
        frame_buffer.append(initial_state)

        while not done and self.frame_count < max_frames:
            self.switch_mode()
            
            # 從 buffer 獲取 s_t 和 s_t+1
            s_t, s_t_plus_1 = list(frame_buffer)
            
            # 決定動作 a_t
            action = self.choose_action()
            
            # 執行動作，獲取獎勵 r_t+1 和下一幀 s_t+2
            s_t_plus_2, reward, done = self.env.step(action)
            
            # <<< 核心修改：降采样邏輯 >>>
            # 檢查獎勵是否是低信息量的
            is_low_info_reward = abs(reward) < REWARD_THRESHOLD
            
            # 如果是低信息量獎勵，則按機率丟棄
            if is_low_info_reward and random.random() < DISCARD_PROBABILITY:
                # 丟棄數據，什麼都不做
                pass
            else:
                # 保留數據
                self.dataset['s_t'].append(s_t)
                self.dataset['s_t_plus_1'].append(s_t_plus_1)
                self.dataset['a_t'].append(action)
                self.dataset['r_t_plus_1'].append(reward)

            # 將新的一幀加入 buffer，準備下一次循環
            frame_buffer.append(s_t_plus_2)
            self.frame_count += 1
            
            # 渲染畫面
            self.env.render()

        print(f"Episode finished. Kept {len(self.dataset['a_t'])} data points so far.")

    def save_data(self, path):
        # ... (save_data 邏輯保持不變) ...
        if not self.dataset['a_t']:
            print("No data collected. Skipping save.")
            return
        processed_dataset = {}
        for key, value in self.dataset.items():
            if key.startswith('s_'): dtype = np.uint8
            elif key.startswith('a_'): dtype = np.int16
            elif key.startswith('r_'): dtype = np.float32
            else: dtype = None 
            processed_dataset[key] = np.array(value, dtype=dtype)
        np.savez_compressed(path, **processed_dataset)
        print(f"Dataset saved to {path}. Final size: {len(processed_dataset['a_t'])} samples.")


if __name__ == '__main__':
    env = DinoGameEnv(render_mode='human')
    collector = RewardDataCollector(env)

    for i in range(NUM_EPISODES_TO_COLLECT):
        print(f"\n--- Starting Episode {i + 1}/{NUM_EPISODES_TO_COLLECT} ---")
        collector.collect_episode(MAX_FRAMES_PER_EPISODE)

    collector.save_data(DATASET_PATH)

    pygame.quit()