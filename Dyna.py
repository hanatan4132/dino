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
from train import ImagePredictionNet
from trainreward import RewardPredictor

# --- 2. Dyna-Q 的世界模型 ---
class WorldModel:
    def __init__(self, image_model_path, reward_model_path, device):
        self.device = device
        self.image_predictor = ImagePredictionNet(action_size=3).to(self.device)
        self.image_predictor.load_state_dict(torch.load(image_model_path, map_location=self.device))
        self.image_predictor.eval()
        self.reward_predictor = RewardPredictor(action_size=3).to(self.device)
        self.reward_predictor.load_state_dict(torch.load(reward_model_path, map_location=self.device))
        self.reward_predictor.eval()
        print("World Model (Image & Reward Predictors) loaded successfully.")
    def preprocess_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)
        # 增加 channel 維度
        
        return frame  # 保持 numpy array，不轉 torch.Tensor
    def predict(self, s_t_frame, s_t_plus_1_frame, action_tensor):
        s_t_frame=s_t_frame.unsqueeze(0)
        s_t_plus_1_frame=s_t_plus_1_frame.unsqueeze(0)
        with torch.no_grad():
            pred_s_t_plus_2, pred_s_t_plus_3 = self.image_predictor(s_t_frame, s_t_plus_1_frame, action_tensor)
            pred_reward = self.reward_predictor(s_t_frame, s_t_plus_1_frame, action_tensor).item()
        return pred_s_t_plus_2, pred_s_t_plus_3, pred_reward

# --- 3. DDQN Agent 和 ReplayBuffer (與純粹DDQN版本完全相同) ---
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
        # StepLR 會每隔 'step_size' 個 epoch/step，將學習率乘以 'gamma'
        self.real_scheduler = optim.lr_scheduler.StepLR(self.real_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.virtual_scheduler = optim.lr_scheduler.StepLR(self.virtual_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
        print(f"Agent initialized. Real LR: {real_lr}, Virtual LR: {virtual_lr}")
        print(f"Gradient Clipping: {self.gradient_clip_value}, LR Scheduler Step: {scheduler_step_size}, Gamma: {scheduler_gamma}")
    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
                return self.policy_net(state).max(1)[1].item()
        else: return random.randrange(self.num_actions)
    def learn(self, is_virtual=False):
        """根據經驗來源，使用對應的 memory, batch_size 和 optimizer 進行學習"""
        memory = self.virtual_memory if is_virtual else self.real_memory
        batch_size = self.virtual_batch_size if is_virtual else self.real_batch_size
        optimizer = self.virtual_optimizer if is_virtual else self.real_optimizer
        
        if len(memory) < batch_size: return None
        
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
        
        optimizer.zero_grad(); 
        loss.backward(); 
        
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_value)
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
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)
    def __len__(self): return len(self.buffer)
    
# --- 4. 輔助函式 ---
def sanitize_filename(name): return re.sub(r'[\\/*?:"<>|]', "_", name)
def save_experiment_results(mean_scores, std_scores, filename_prefix="experiment"):
    # ... (儲存 JSON 的函式不變)
    result = {"mean": mean_scores.tolist(), "std": std_scores.tolist()}
    os.makedirs("experiment_results", exist_ok=True); timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/{filename_prefix}_{timestamp}.json"
    with open(filename, "w") as f: json.dump(result, f, indent=2)
    print(f"✅ 實驗結果已儲存到 {filename}")
def show_tensor_img(tensor, title):
    img = tensor.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
# --- 5. 核心實驗函式 (Dyna-Q 版本) ---
def run_dynaq_experiment(exp_name, world_model, planning_steps, plot_freq=50, averaging_window=10):
    env = DinoGameEnv(render_mode='huma')
    input_shape = (STACK_SIZE, env.observation_shape[0], env.observation_shape[1])
    agent = DDQNAgent(
        input_shape, env.action_space, BUFFER_SIZE, 
        BATCH_SIZE_REAL, BATCH_SIZE_VIRTUAL, 
        GAMMA, LEARNING_RATE_REAL, LEARNING_RATE_VIRTUAL, TAU,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        scheduler_step_size=SCHEDULER_STEP_SIZE,
        scheduler_gamma=SCHEDULER_GAMMA
    )
    averaged_scores = []; temp_scores_window = []
    epsilon = EPSILON_START; total_steps = 0
    pp = 0
    pe=0
    pn = 0
    rp = 0
    re = 0
    for episode in range(1, TOTAL_EPISODES + 1):
        # 初始化/重置
        frame_t_minus_1 = env.reset()
        frame_t, _, _ = env.step(0)
        frame_t_minus_1 = world_model.preprocess_frame(frame_t_minus_1)
        frame_t = world_model.preprocess_frame(frame_t)
        current_state_np = np.stack([frame_t_minus_1, frame_t], axis=1)
        episode_score = 0
        steps = 0
        nodes = []
        prediction_tree = [None] * 10
        current_position = 0
        while True:
            # 1. 真實互動

            action = agent.select_action(current_state_np.squeeze(0), epsilon)
            
            next_frame_1, reward, done = env.step(action)

            next_frame_1 = world_model.preprocess_frame(next_frame_1)
            frame_t_minus_1 = frame_t
            frame_t = next_frame_1

            actio = agent.select_action(current_state_np.squeeze(0), epsilon)
            next_frame_2, _, _ = env.step(actio) 

            next_frame_2 = world_model.preprocess_frame(next_frame_2)
            frame_t_minus_1 = frame_t
            frame_t = next_frame_2
            #print(current_state_np[0].shape,current_state_np[1].shape)
            #print(next_frame_1.shape,next_frame_2.shape)
            #env.render()
            
            next_state_np = np.stack([frame_t_minus_1, frame_t], axis=1)
            total_steps+=2
            # <<< 將真實經驗存入 real_memory >>>

            agent.store_experience(current_state_np, action, reward, next_state_np, done, is_virtual=False)
            re+=1
            if reward > 6:
                rp+=1
            # 2. 從真實經驗中學習
            agent.learn(is_virtual=False)
            # a. 每 4 步，用當前的真實狀態重置規劃的起點
            current_state_np = next_state_np
            #f reward > 0:
            episode_score = env.points
            
            
            
            if steps % 5 == 0 :
                tensor_1 = torch.FloatTensor(frame_t_minus_1).to(agent.device)
                tensor_2 = torch.FloatTensor(frame_t).to(agent.device)
                nodes = [(tensor_1, tensor_2)]
      
            new_nodes = []
            for s2_tensor, s1_tensor in nodes:
                for act in range(agent.num_actions):
                    # a. 使用世界模型進行預測
                    act_tensor = torch.LongTensor([[act]]).to(agent.device)

                    pred_st1p, pred_st2p, pred_reward = world_model.predict(s2_tensor, s1_tensor, act_tensor)
                    imagined_done = pred_reward < -40
                    
                    
                    seed = random.random()
                    if -15 < pred_reward < 2 and seed > 0.1:
                            #print("seed:",seed)
                            continue
                    #print("pred_reward seed",pred_reward,seed)
                    
                    if pred_reward > 4: 
                        pp += 1
                        
                    if pred_reward > 6:
                        pred_reward = 10    
                    agent.store_experience(
                        np.stack([s2_tensor.cpu().numpy(), s1_tensor.cpu().numpy()], axis=1),
                        act,
                        pred_reward,
                        np.stack([pred_st1p.squeeze(0).cpu().numpy(), pred_st2p.squeeze(0).cpu().numpy()], axis=1),
                        imagined_done,
                        is_virtual=True
                    )
                    pe+=1
                    # c. 將預測出的新幀對加入 new_nodes
                    new_nodes.append((pred_st1p.squeeze(0), pred_st2p.squeeze(0)))
                    
           
            # 更新節點列表
            nodes = new_nodes
            
            
            '''
            tensor_1 = torch.FloatTensor(frame_t_minus_1).to(agent.device)
            tensor_2 = torch.FloatTensor(frame_t).to(agent.device)
            
            for i in range(10):
                # 隨機選擇動作
                action = random.randint(0, agent.num_actions - 1)
                action_tensor = torch.LongTensor([[action]]).to(agent.device)
                pe+=1
                # 預測
                pred_st1p, pred_st2p, pred_reward = world_model.predict(tensor_1, tensor_2, action_tensor)
                imagined_done = pred_reward < -40
                if pred_reward > 4:
                    pp +=1
                    show_tensor_img(pred_st1p, pred_reward)
                    show_tensor_img(pred_st2p, pred_reward)
                    
                    print("pe pp ",pe,pp)
                elif pred_reward < -40:
                    pn+=1
                    print("pn",pn)
                agent.store_experience(
                    np.stack([tensor_1.squeeze().cpu().numpy(), tensor_2.squeeze().cpu().numpy()], axis=0),
                    action,
                    pred_reward,
                    np.stack([pred_st1p.squeeze().cpu().numpy(), pred_st2p.squeeze().cpu().numpy()], axis=0),
                    imagined_done,
                    is_virtual=True
                )

                tensor_1, tensor_2 = pred_st1p.squeeze(0), pred_st2p.squeeze(0)
            '''
             
            
            '''
            tensor_1 = torch.FloatTensor(frame_t_minus_1).to(agent.device)
            tensor_2 = torch.FloatTensor(frame_t).to(agent.device)
            prediction_tree[current_position] = (tensor_1, tensor_2, None, None)
            
            # b. 更新所有已存在的預測路徑
            with torch.no_grad():
                for idx in range(len(prediction_tree)):
                    if prediction_tree[idx] is not None:
                        # 從樹中取出一個想像的起點
                        p_prev_frame_2, p_prev_frame_1, _, _ = prediction_tree[idx]
                        
                        # <<< 關鍵：如何選擇動作？ >>>
                        # 您的邏輯中沒有明確指出這裡用什麼動作。
                        # 最簡單的方式是隨機選擇，以增加探索。
                        imagined_action = random.randint(0, agent.num_actions - 1)
                        action_tensor = torch.LongTensor([[imagined_action]]).to(agent.device)

                        # 使用世界模型預測下一個狀態和獎勵
                        pred_st1p, pred_st2p, pred_reward = world_model.predict(p_prev_frame_2, p_prev_frame_1, action_tensor)
                        imagined_done = pred_reward < -40
                        # 將預測出的虛擬經驗存入虛擬經驗池
                        agent.store_experience(
                            np.stack([p_prev_frame_2.squeeze().cpu().numpy(), p_prev_frame_1.squeeze().cpu().numpy()], axis=0),
                            imagined_action,
                            pred_reward,
                            np.stack([pred_st1p.squeeze().cpu().numpy(), pred_st2p.squeeze().cpu().numpy()], axis=0),
                            imagined_done,
                            is_virtual=True
                        )

                        # 更新預測樹中的這條路徑，讓它向前"生長"一步
                        prediction_tree[idx] = (pred_st1p.squeeze(0), pred_st2p.squeeze(0), None, None)

            # c. 更新當前位置指針，循環使用預測樹
            current_position = (current_position + 1) % 10
            '''
            
      
            
            # --- 4. 從虛擬經驗中學習 ---
            if total_steps % 20 == 0:
                agent.learn(is_virtual=True)
        
            
            if done: break
        epsilon = max(EPSILON_END, EPSILON_START - total_steps / EPSILON_DECAY)      
        if episode % 5 == 0: 
            #print("upadate")
            agent.update_target_network()
        # 分數記錄
        #agent.step_schedulers()
        
        # 顯示當前學習率，方便觀察
        current_lr_real = agent.real_optimizer.param_groups[0]['lr']
        
        #print(f"Exp: {exp_name} | Eps {episode} | Score: {episode_score:.2f} | Epsilon: {epsilon:.4f} | Real LR: {current_lr_real:.7f}")
        temp_scores_window.append(episode_score)
        if episode % averaging_window == 0:
            avg_score = np.mean(temp_scores_window); averaged_scores.append(avg_score); temp_scores_window = []
            print(f"Exp: {exp_name} | Eps {episode-averaging_window+1}-{episode} | Avg Score: {avg_score:.2f} | Epsilon: {epsilon:.4f} | (pe,pp):{pe} , {pp}| (re,rp):{re} , {rp}")
            if episode % plot_freq == 0 and len(averaged_scores) > 0:
                 fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(averaged_scores)
                 
                 ax.set_title(f"Live Avg Scores for {exp_name} (Episode {episode})")
                 ax.set_xlabel(f"Episode Window (x{averaging_window})"); ax.set_ylabel(f"Avg Score per {averaging_window} eps")
                 ax.grid(True); plt.show(block=False); plt.pause(0.1); plt.close(fig)

    pygame.quit()
    return averaged_scores


EXPERIMENT_NAME = "DynaQ_Dinox deepb5 03 20 trash0.2"

GRADIENT_CLIP_VALUE = 500 # 梯度裁剪的範數上限，1.0 是個常用的好起點
SCHEDULER_STEP_SIZE = 200 # 每 1000 個 episodes，學習率衰減一次
SCHEDULER_GAMMA = 0.9
NUM_RUNS = 3; TOTAL_EPISODES = 1001
AVERAGING_WINDOW = 10; PLOT_FREQUENCY = 100
PLANNING_STEPS = 5 # <<< Dyna-Q 的核心參數

# DDQN 超參數
STACK_SIZE = 2 # <<< 必須是 2
GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY = 0.99, 1.0, 0.01, 7000
BUFFER_SIZE, TARGET_UPDATE_FREQ = 6000, 250
TAU =  1e-3
# <<< 新的、分離的超參數 >>>
BATCH_SIZE_REAL = 64
BATCH_SIZE_VIRTUAL = 32 # 虛擬經驗的 batch size 可以不同
LEARNING_RATE_REAL = 0.0001
LEARNING_RATE_VIRTUAL = 0.00003# 虛擬經驗使用更小的學習率
# 模型路徑
IMAGE_MODEL_PATH = "dino_world_models/best_image_model.pth"
REWARD_MODEL_PATH = "dino_reward_models/best_reward_model.pth"

if __name__ == '__main__':
    if not os.path.exists(IMAGE_MODEL_PATH) or not os.path.exists(REWARD_MODEL_PATH):
        print("Error: World model files not found. Please train them first.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_model = WorldModel(IMAGE_MODEL_PATH, REWARD_MODEL_PATH, device)
        
        all_runs_scores = []
        safe_exp_name = sanitize_filename(EXPERIMENT_NAME)

        for i in range(NUM_RUNS):
            print(f"\n{'='*15} Starting Run {i + 1}/{NUM_RUNS} {'='*15}")
            scores_for_one_run = run_dynaq_experiment(f"Run {i+1}", world_model, PLANNING_STEPS, PLOT_FREQUENCY, AVERAGING_WINDOW)
            all_runs_scores.append(scores_for_one_run)
        
        print(f"\n{'='*15} All runs completed. Processing results... {'='*15}")
        min_len = min(len(scores) for scores in all_runs_scores)
        aligned_scores = np.array([scores[:min_len] for scores in all_runs_scores])
        mean_scores, std_scores = np.mean(aligned_scores, axis=0), np.std(aligned_scores, axis=0)
        save_experiment_results(mean_scores, std_scores, filename_prefix=safe_exp_name)
        
        episodes_axis = np.arange(min_len) * AVERAGING_WINDOW
        plt.figure(figsize=(12, 7)); plt.plot(episodes_axis, mean_scores, label=f'Mean Score (over {NUM_RUNS} runs)', color='blue')
        plt.fill_between(episodes_axis, mean_scores - std_scores, mean_scores + std_scores, color='blue', alpha=0.2, label='±1 Std Dev')
        plt.title(f"Dyna-Q DDQN Performance on Dino Game\n({EXPERIMENT_NAME})"); plt.xlabel("Episode"); plt.ylabel(f"Avg Score per {AVERAGING_WINDOW} Episodes")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(f"experiment_results/{safe_exp_name}_plot.png"); plt.show()