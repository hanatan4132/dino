import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import re
import json
import time
import pygame

# --- 引入模組 ---
from dino import DinoGameEnv
from dqn import DQN_Network # 確保你的 DQN 模型輸入通道數是 2

# --- 1. ReplayBuffer 和 DDQNAgent 類 (與之前版本相同，它們是通用的) ---
class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)
    def __len__(self): return len(self.buffer)

class DDQNAgent:
    def __init__(self, input_shape, num_actions, buffer_size, batch_size, gamma, lr, tau):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.num_actions, self.batch_size, self.gamma, self.tau = num_actions, batch_size, gamma, tau
        self.policy_net = DQN_Network(input_shape, num_actions).to(self.device); self.target_net = DQN_Network(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr); self.memory = ReplayBuffer(buffer_size)
    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device); return self.policy_net(state).max(1)[1].item()
        else: return random.randrange(self.num_actions)
    def learn(self):
        if len(self.memory) < self.batch_size: return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device); actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device); next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device); next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions); expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        q_values = self.policy_net(states).gather(1, actions); loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); return loss.item()
    def update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

# --- 2. 輔助函式 (frame_stack 不再需要) ---
# 我們將在主迴圈中直接處理
def sanitize_filename(name): return re.sub(r'[\\/*?:"<>|]', "_", name)
def save_experiment_results(mean_scores, std_scores, filename_prefix="experiment"):
    result = {"mean": mean_scores.tolist(), "std": std_scores.tolist()}
    os.makedirs("experiment_results", exist_ok=True); timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/{filename_prefix}_{timestamp}.json"
    with open(filename, "w") as f: json.dump(result, f, indent=2)
    print(f"✅ 實驗結果已儲存到 {filename}")

# --- 3. 核心實驗函式 (2幀狀態DDQN版本) ---
def run_experiment(exp_name, plot_freq=50, averaging_window=10):
    env = DinoGameEnv(render_mode='rgb_array')
    # <<< 核心修改：Agent 的輸入 shape 現在是 2 幀 >>>
    input_shape = (STACK_SIZE, env.observation_shape[0], env.observation_shape[1])
    agent = DDQNAgent(input_shape, env.action_space, BUFFER_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE, TAU)
    
    averaged_scores = []; temp_scores_window = []
    epsilon = EPSILON_START; total_steps = 0
    plt.ion(); fig, ax = plt.subplots(figsize=(10, 5))
    
    for episode in range(1, TOTAL_EPISODES + 1):
        # --- 初始化/重置 ---
        frame_t_minus_1 = env.reset().squeeze()
        frame_t, _, _ = env.step(0)
        frame_t = frame_t.squeeze()
        current_state_np = np.stack([frame_t_minus_1, frame_t], axis=0)
        episode_score = 0
        
        while True:
            # 1. 根據當前狀態 s_t = (f_{t-1}, f_t) 選擇動作
            action = agent.select_action(current_state_np, epsilon)
            
            # 2. 連續執行兩次相同的動作，得到 f_{t+1} 和 f_{t+2}
            next_frame_1, reward_1, done_1 = env.step(action)
            next_frame_2, reward_2, done_2 = (env.step(action) if not done_1 else (next_frame_1, 0, True))
            done = done_1 or done_2
            final_reward = reward_2 if abs(reward_2) > 1.0 else (reward_1 if abs(reward_1) > 1.0 else reward_1)
            
            # 3. 構建新的狀態 s_{t+2} = (f_{t+1}, f_{t+2})
            next_state_np = np.stack([next_frame_1.squeeze(), next_frame_2.squeeze()], axis=0)
            
            # 4. 儲存經驗 (s_t, a_t, r, s_{t+2}, done)
            agent.memory.push(current_state_np, action, final_reward, next_state_np, done)
            
            # 5. 學習
            agent.learn()
            
            # 6. 更新狀態和參數
            current_state_np = next_state_np
            episode_score = env.points
            total_steps += 2
            epsilon = max(EPSILON_END, EPSILON_START - total_steps / EPSILON_DECAY)
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            if done:
                break
        
        # --- 分數記錄與繪圖 ---
        temp_scores_window.append(episode_score)
        if episode % averaging_window == 0:
            avg_score = np.mean(temp_scores_window); averaged_scores.append(avg_score); temp_scores_window = []
            print(f"Exp: {exp_name} | Eps {episode-averaging_window+1}-{episode} | Avg Score: {avg_score:.2f} | Epsilon: {epsilon:.4f}")
            if episode % plot_freq == 0 and len(averaged_scores) > 0:
                ax.clear(); ax.plot(averaged_scores)
                ax.set_title(f"Live Avg Scores for {exp_name} (Episode {episode})")
                ax.set_xlabel(f"Episode Window (x{averaging_window})"); ax.set_ylabel(f"Avg Score per {averaging_window} eps")
                ax.grid(True); plt.draw(); plt.pause(0.01)

    plt.ioff(); plt.close(fig); pygame.quit()
    return averaged_scores

# --- 4. 主執行區塊 (修改 STACK_SIZE) ---
# --- 超參數與實驗設定 ---
EXPERIMENT_NAME = "DDQN_Dino_2FrameState_Corrected"
NUM_RUNS = 5
TOTAL_EPISODES = 500
AVERAGING_WINDOW = 10
PLOT_FREQUENCY = 50

# <<< 核心修改：STACK_SIZE 現在是 2 >>>
STACK_SIZE = 2
# DDQN 超參數
GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY = 0.99, 1.0, 0.01, 100000
BUFFER_SIZE, BATCH_SIZE, TARGET_UPDATE_FREQ = 50000, 64, 1000
LEARNING_RATE, TAU = 1e-4, 1e-3

if __name__ == '__main__':
    all_runs_scores = []
    safe_exp_name = sanitize_filename(EXPERIMENT_NAME)

    for i in range(NUM_RUNS):
        print(f"\n{'='*15} Starting Run {i + 1}/{NUM_RUNS} {'='*15}")
        scores_for_one_run = run_experiment(f"Run {i+1}", plot_freq=PLOT_FREQUENCY, averaging_window=AVERAGING_WINDOW)
        all_runs_scores.append(scores_for_one_run)
    
    print(f"\n{'='*15} All runs completed. Processing results... {'='*15}")

    min_len = min(len(scores) for scores in all_runs_scores)
    aligned_scores = np.array([scores[:min_len] for scores in all_runs_scores])
    mean_scores, std_scores = np.mean(aligned_scores, axis=0), np.std(aligned_scores, axis=0)

    save_experiment_results(mean_scores, std_scores, filename_prefix=safe_exp_name)

    # --- 繪製最終的統計結果圖 ---
    episodes_axis = np.arange(min_len) * AVERAGING_WINDOW
    plt.figure(figsize=(12, 7)); plt.plot(episodes_axis, mean_scores, label=f'Mean Score (over {NUM_RUNS} runs)', color='blue')
    plt.fill_between(episodes_axis, mean_scores - std_scores, mean_scores + std_scores, color='blue', alpha=0.2, label='±1 Standard Deviation')
    plt.title(f"DDQN Performance on Dino Game\n({EXPERIMENT_NAME})"); plt.xlabel("Episode")
    plt.ylabel(f"Average Score per {AVERAGING_WINDOW} Episodes"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"experiment_results/{safe_exp_name}_plot.png"); plt.show()