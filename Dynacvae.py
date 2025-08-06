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

# --- 1. 引入所有必要的模組 ---
from dino import DinoGameEnv
from dqn import DQN_Network
from trainvae import ConditionalVAE

# --- 2. VAE 世界模型類 ---
class WorldModel:
    def __init__(self, model_path, device):
        """
        使用訓練好的 ConditionalVAE 作為世界模型
        
        Args:
            model_path (str): 訓練好的 VAE 模型路徑
            device: torch device
        """
        self.device = device
        
        # 載入訓練好的 ConditionalVAE 模型
        self.vae_model = ConditionalVAE(
            latent_dim=64,  # 與訓練時保持一致
            action_dim=1,
            in_channels=2,
            out_channels=2
        ).to(device)
        
        # 載入模型權重
        if os.path.exists(model_path):
            self.vae_model.load_state_dict(torch.load(model_path, map_location=device))
            self.vae_model.eval()  # 設為評估模式
            print(f"World model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def preprocess_frame(self, frame):
        """
        前處理單一畫面，與訓練時的處理方式保持一致
        
        Args:
            frame: numpy array of shape (H, W) or (H, W, 1)
        
        Returns:
            numpy array of shape (H, W) normalized to [0, 1]
        """
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame.squeeze(2)  # 移除最後一個維度
        
        # 正規化到 [0, 1] 範圍
        frame_normalized = frame.astype(np.float32) / 255.0
        return frame_normalized
    
    def predict(self, s_t_minus_1, s_t, action):
        """
        使用 VAE 預測下一個狀態和獎勵
        
        Args:
            s_t_minus_1: torch.Tensor of shape (H, W) - 前一幀
            s_t: torch.Tensor of shape (H, W) - 當前幀  
            action: torch.Tensor of shape (1, 1) - 動作
        
        Returns:
            pred_s_t_plus_1: torch.Tensor - 預測的下一幀
            pred_s_t_plus_2: torch.Tensor - 預測的再下一幀
            pred_reward: float - 預測的獎勵
        """
        self.vae_model.eval()
        
        with torch.no_grad():
            # 確保輸入張量有正確的維度
            if s_t_minus_1.dim() == 2:
                s_t_minus_1 = s_t_minus_1.unsqueeze(0)  # 添加 channel 維度
            if s_t.dim() == 2:
                s_t = s_t.unsqueeze(0)  # 添加 channel 維度
            
            # 堆疊兩幀作為輸入 (1, 2, H, W)
            input_stack = torch.cat([s_t_minus_1, s_t], dim=0).unsqueeze(0)
            
            # 確保動作張量有正確的形狀
            if action.dim() == 1:
                action = action.unsqueeze(0)  # (1, 1)
            
            # 使用 VAE 進行預測
            pred_frames, pred_reward_tensor, _, _ = self.vae_model(input_stack, action)
            
            # 提取預測的兩幀
            pred_s_t_plus_1 = pred_frames[0, 0, :, :]  # 第一幀
            pred_s_t_plus_2 = pred_frames[0, 1, :, :]  # 第二幀
            
            # 提取預測的獎勵
            pred_reward = pred_reward_tensor[0, 0].item()
            
            return pred_s_t_plus_1, pred_s_t_plus_2, pred_reward

# --- 3. DDQN Agent 和 ReplayBuffer (與原版相同) ---
class DDQNAgent:
    def __init__(self, input_shape, num_actions, buffer_size, real_batch_size, virtual_batch_size, gamma, real_lr, virtual_lr, tau,
                 gradient_clip_value=1.0, scheduler_step_size=1000, scheduler_gamma=0.9):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions, self.gamma, self.tau = num_actions, gamma, tau
        
        # Batch sizes
        self.real_batch_size = real_batch_size
        self.virtual_batch_size = virtual_batch_size

        # --- 新增：梯度裁剪值 ---
        self.gradient_clip_value = gradient_clip_value

        # 網路 (共用一套Q網路和目標網路)
        self.policy_net = DQN_Network(input_shape, num_actions).to(self.device)
        self.target_net = DQN_Network(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.real_memory = ReplayBuffer(buffer_size)
        self.virtual_memory = ReplayBuffer(buffer_size)
        
        self.real_optimizer = optim.Adam(self.policy_net.parameters(), lr=real_lr)
        self.virtual_optimizer = optim.Adam(self.policy_net.parameters(), lr=virtual_lr)
        
        # --- 新增：為每個優化器創建對應的學習率調度器 ---
        self.real_scheduler = optim.lr_scheduler.StepLR(self.real_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.virtual_scheduler = optim.lr_scheduler.StepLR(self.virtual_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
        print(f"Agent initialized. Real LR: {real_lr}, Virtual LR: {virtual_lr}")
        print(f"Gradient Clipping: {self.gradient_clip_value}, LR Scheduler Step: {scheduler_step_size}, Gamma: {scheduler_gamma}")
    
    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
                return self.policy_net(state).max(1)[1].item()
        else: 
            return random.randrange(self.num_actions)
    
    def learn(self, is_virtual=False):
        """根據經驗來源，使用對應的 memory, batch_size 和 optimizer 進行學習"""
        memory = self.virtual_memory if is_virtual else self.real_memory
        batch_size = self.virtual_batch_size if is_virtual else self.real_batch_size
        optimizer = self.virtual_optimizer if is_virtual else self.real_optimizer
        
        if len(memory) < batch_size: 
            return None
        
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        states = torch.FloatTensor(states).squeeze(1).squeeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).squeeze(1).squeeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        q_values = self.policy_net(states).gather(1, actions)
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        optimizer.zero_grad()
        loss.backward()
        
        # 可選的梯度裁剪
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_value)
        optimizer.step()
        return loss.item()
    
    def step_schedulers(self):
        """在每個 episode 結束後調用，以更新學習率"""
        self.real_scheduler.step()
        self.virtual_scheduler.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def store_experience(self, state, action, reward, next_state, done, is_virtual=False):
        """根據經驗來源，存入不同的 memory buffer"""
        if is_virtual:
            self.virtual_memory.push(state, action, reward, next_state, done)
        else:
            self.real_memory.push(state, action, reward, next_state, done)

class ReplayBuffer:
    def __init__(self, capacity): 
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)
    
    def __len__(self): 
        return len(self.buffer)

# --- 4. 輔助函式 ---
def sanitize_filename(name): 
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_experiment_results(mean_scores, std_scores, filename_prefix="experiment"):
    result = {"mean": mean_scores.tolist(), "std": std_scores.tolist()}
    os.makedirs("experiment_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/{filename_prefix}_{timestamp}.json"
    with open(filename, "w") as f: 
        json.dump(result, f, indent=2)
    print(f"✅ 實驗結果已儲存到 {filename}")

def show_tensor_img(tensor, title):
    img = tensor.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- 5. 核心實驗函式 (Dyna-Q VAE 版本) ---
def run_dynaq_experiment(exp_name, world_model, planning_steps, plot_freq=50, averaging_window=10):
    env = DinoGameEnv(render_mode='humaqn')
    input_shape = (STACK_SIZE, env.observation_shape[0], env.observation_shape[1])
    agent = DDQNAgent(
        input_shape, env.action_space, BUFFER_SIZE, 
        BATCH_SIZE_REAL, BATCH_SIZE_VIRTUAL, 
        GAMMA, LEARNING_RATE_REAL, LEARNING_RATE_VIRTUAL, TAU,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        scheduler_step_size=SCHEDULER_STEP_SIZE,
        scheduler_gamma=SCHEDULER_GAMMA
    )
    
    averaged_scores = []
    temp_scores_window = []
    epsilon = EPSILON_START
    total_steps = 0
    
    for episode in range(1, TOTAL_EPISODES + 1):
        # 初始化/重置
        frame_t_minus_1 = env.reset()
        frame_t, _, _ = env.step(0)
        
        frame_t_minus_1 = world_model.preprocess_frame(frame_t_minus_1)
        frame_t = world_model.preprocess_frame(frame_t)
        
        current_state_np = np.stack([frame_t_minus_1, frame_t], axis=0)
        episode_score = 0
        steps = 0
        nodes = []
        
        while True:
            # 1. 真實互動
            action = agent.select_action(current_state_np, epsilon)
            
            next_frame_1, reward, done = env.step(action)

            next_frame_1 = world_model.preprocess_frame(next_frame_1)
            frame_t_minus_1 = frame_t
            frame_t = next_frame_1

            actio = agent.select_action(current_state_np, epsilon)
            next_frame_2, _, _ = env.step(actio) 

            next_frame_2 = world_model.preprocess_frame(next_frame_2)
            frame_t_minus_1 = frame_t
            frame_t = next_frame_2
            #print(current_state_np[0].shape,current_state_np[1].shape)
            #print(next_frame_1.shape,next_frame_2.shape)
            #env.render()
            
            next_state_np = np.stack([frame_t_minus_1, frame_t], axis=0)
            total_steps+=2
            # <<< 將真實經驗存入 real_memory >>>

            agent.store_experience(current_state_np, action, reward, next_state_np, done, is_virtual=False)
            #print(current_state_np.shape, next_state_np.shape)
            # 2. 從真實經驗中學習
            agent.learn(is_virtual=False)
            # a. 每 4 步，用當前的真實狀態重置規劃的起點
            current_state_np = next_state_np
            #f reward > 0:
            episode_score = env.points
            
            # 3. 使用 VAE 世界模型進行規劃
            if steps % 5 == 0:
                # 將當前狀態轉換為 tensor
                tensor_1 = torch.FloatTensor(frame_t_minus_1).to(agent.device)
                tensor_2 = torch.FloatTensor(frame_t).to(agent.device)
                nodes = [(tensor_1, tensor_2)]
            
            # 使用節點進行多步規劃
            new_nodes = []
            for s_t_minus_1_tensor, s_t_tensor in nodes:
                for act in range(agent.num_actions):
                    # 使用 VAE 世界模型進行預測
                    act_tensor = torch.LongTensor([[act]]).to(agent.device)
                   
                    pred_s_t_plus_1, pred_s_t_plus_2, pred_reward = world_model.predict(
                        s_t_minus_1_tensor, s_t_tensor, act_tensor
                        )
                    
                    # 判斷是否結束（基於獎勵）
                    imagined_done = pred_reward < -60
                    seed = random.random()
                    if -15 < pred_reward < 2 and seed > 0.05:
                            #print("seed:",seed)
                            continue
                        
                    # 將虛擬經驗存入 virtual_memory
                    agent.store_experience(
                        np.stack([s_t_minus_1_tensor.cpu().numpy(), s_t_tensor.cpu().numpy()], axis=0),
                        act,
                        pred_reward,
                        np.stack([pred_s_t_plus_1.cpu().numpy(), pred_s_t_plus_2.cpu().numpy()], axis=0),
                        imagined_done,
                        is_virtual=True
                    )
                    
                    # 將預測出的新幀對加入 new_nodes
                    new_nodes.append((pred_s_t_plus_1, pred_s_t_plus_2))
            
            # 更新節點列表
            nodes = new_nodes
            
            
            '''
            tensor_1 = torch.FloatTensor(frame_t_minus_1).to(agent.device)
            tensor_2 = torch.FloatTensor(frame_t).to(agent.device)
            
            for i in range(15):
                # 隨機選擇動作
                act = random.randint(0, agent.num_actions - 1)
                act_tensor = torch.LongTensor([[act]]).to(agent.device)
               
                pred_s_t_plus_1, pred_s_t_plus_2, pred_reward = world_model.predict(
                    tensor_1, tensor_2, act_tensor
                    )
                
                # 判斷是否結束（基於獎勵）
                imagined_done = pred_reward < -60
                seed = random.random()
                if -15 < pred_reward < 2 and seed > 0.05:
                        #print("seed:",seed)
                        continue
                    
                # 將虛擬經驗存入 virtual_memory
                agent.store_experience(
                    np.stack([tensor_1.cpu().numpy(), tensor_2.cpu().numpy()], axis=0),
                    act,
                    pred_reward,
                    np.stack([pred_s_t_plus_1.cpu().numpy(), pred_s_t_plus_2.cpu().numpy()], axis=0),
                    imagined_done,
                    is_virtual=True
                )

                tensor_1, tensor_2 = pred_s_t_plus_1.squeeze(0), pred_s_t_plus_2.squeeze(0)

            '''
            # 4. 從虛擬經驗中學習
            if total_steps % 20 == 0:
                agent.learn(is_virtual=True)
            
            if done: 
                break
        
        # 更新 epsilon
        epsilon = max(EPSILON_END, EPSILON_START - total_steps / EPSILON_DECAY)
        
        # 定期更新目標網路
        if episode % 5 == 0: 
            agent.update_target_network()
        
        # 分數記錄和顯示
        current_lr_real = agent.real_optimizer.param_groups[0]['lr']
        
        temp_scores_window.append(episode_score)
        if episode % averaging_window == 0:
            avg_score = np.mean(temp_scores_window)
            averaged_scores.append(avg_score)
            temp_scores_window = []
            print(f"Exp: {exp_name} | Eps {episode-averaging_window+1}-{episode} | Avg Score: {avg_score:.2f} | Epsilon: {epsilon:.4f}")
            
            if episode % plot_freq == 0 and len(averaged_scores) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(averaged_scores)
                ax.set_title(f"Live Avg Scores for {exp_name} (Episode {episode})")
                ax.set_xlabel(f"Episode Window (x{averaging_window})")
                ax.set_ylabel(f"Avg Score per {averaging_window} eps")
                ax.grid(True)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close(fig)

    pygame.quit()
    return averaged_scores

# --- 6. 主要參數設定 ---
EXPERIMENT_NAME = "DynaQ_VAE_Dino_deepb 20 03 trash0.05"

GRADIENT_CLIP_VALUE = 500
SCHEDULER_STEP_SIZE = 200
SCHEDULER_GAMMA = 0.9
NUM_RUNS = 3
TOTAL_EPISODES = 1001
AVERAGING_WINDOW = 10
PLOT_FREQUENCY = 100
PLANNING_STEPS = 5

# DDQN 超參數
STACK_SIZE = 2
GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY = 0.99, 1.0, 0.01, 7000
BUFFER_SIZE, TARGET_UPDATE_FREQ = 6000, 250
TAU = 1e-3

# 分離的超參數
BATCH_SIZE_REAL = 64
BATCH_SIZE_VIRTUAL = 32
LEARNING_RATE_REAL = 0.0001
LEARNING_RATE_VIRTUAL = 0.00003

# VAE 模型路徑 (使用訓練好的最佳模型)
VAE_MODEL_PATH = "checkpoints_vae/best_cvae_model.pth"

if __name__ == '__main__':
    if not os.path.exists(VAE_MODEL_PATH):
        print(f"Error: VAE model file not found at {VAE_MODEL_PATH}. Please train the VAE first using trainvae.py")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_model = WorldModel(VAE_MODEL_PATH, device)
        
        all_runs_scores = []
        safe_exp_name = sanitize_filename(EXPERIMENT_NAME)

        for i in range(NUM_RUNS):
            print(f"\n{'='*15} Starting Run {i + 1}/{NUM_RUNS} {'='*15}")
            scores_for_one_run = run_dynaq_experiment(
                f"Run {i+1}", world_model, PLANNING_STEPS, 
                PLOT_FREQUENCY, AVERAGING_WINDOW
            )
            all_runs_scores.append(scores_for_one_run)
        
        print(f"\n{'='*15} All runs completed. Processing results... {'='*15}")
        min_len = min(len(scores) for scores in all_runs_scores)
        aligned_scores = np.array([scores[:min_len] for scores in all_runs_scores])
        mean_scores, std_scores = np.mean(aligned_scores, axis=0), np.std(aligned_scores, axis=0)
        save_experiment_results(mean_scores, std_scores, filename_prefix=safe_exp_name)
        
        episodes_axis = np.arange(min_len) * AVERAGING_WINDOW
        plt.figure(figsize=(12, 7))
        plt.plot(episodes_axis, mean_scores, label=f'Mean Score (over {NUM_RUNS} runs)', color='blue')
        plt.fill_between(episodes_axis, mean_scores - std_scores, mean_scores + std_scores, 
                        color='blue', alpha=0.2, label='±1 Std Dev')
        plt.title(f"Dyna-Q VAE DDQN Performance on Dino Game\n({EXPERIMENT_NAME})")
        plt.xlabel("Episode")
        plt.ylabel(f"Avg Score per {AVERAGING_WINDOW} Episodes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"experiment_results/{safe_exp_name}_plot.png")
        plt.show()