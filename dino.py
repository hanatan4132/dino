import pygame
import os
import random
import numpy as np
import cv2
import time
# --- 遊戲設定 (保持不變) ---
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
# 為了 AI 訓練，可以選擇隱藏畫面以加速
# SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) 

# 載入圖片資源 (保持不變)
RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]
SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]
BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]
CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


# --- 遊戲物件類別 (稍微修改以移除全域變數依賴) ---
class Dinosaur:
    X_POS = 80
    Y_POS = 300
    Y_POS_DUCK = 330
    JUMP_VEL = 8

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False
        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    # userInput 現在是一個數字 (0: run, 1: jump, 2: duck)
    def update(self, action):
        # 狀態更新邏輯
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        # === 修改後的 AI 動作輸入邏輯 ===
        # 如果不在跳躍過程中
        if not self.dino_jump:
            if action == 1: # 指令: 跳躍
                self.dino_duck = False
                self.dino_run = False
                self.dino_jump = True
            elif action == 2: # 指令: 蹲下
                self.dino_duck = True
                self.dino_run = False
                self.dino_jump = False
            elif action == 0: # 指令: 跑步 (或無指令)
                self.dino_duck = False # <<< 關鍵修改：強制取消蹲下狀態
                self.dino_run = True
                self.dino_jump = False
    
    # 以下 duck, run, jump, draw 方法不變...
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
       self.image = self.jump_img
       if self.dino_jump:
           self.dino_rect.y -= self.jump_vel * 4
           self.jump_vel -= 0.8
       if self.jump_vel < - self.JUMP_VEL:
           self.dino_jump = False
           self.jump_vel = self.JUMP_VEL


    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cloud:
    # ... (Cloud class code, unchanged) ...
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    # ... (Obstacle and its subclasses, slightly modified) ...
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.passed = False 
    def update(self, game_speed):
        self.rect.x -= game_speed

    def draw(self, SCREEN, game_speed):
        draw_x = self.rect.x - game_speed
        draw_y = self.rect.y
        SCREEN.blit(self.image[self.type], (draw_x, draw_y))

class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 260
        self.index = 0


    def draw(self, SCREEN, game_speed):
        if self.index >= 9:
            self.index = 0
        draw_x = self.rect.x - game_speed
        draw_y = self.rect.y
        SCREEN.blit(self.image[self.index//5], (draw_x, draw_y))
        self.index += 1


# --- 主要的 RL 環境類別 ---
class DinoGameEnv:
    def __init__(self, render_mode='human'):
        pygame.init()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else: # 'rgb_array' or None
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.action_space = 2  # 0: run, 1: jump, 2: duck
        self.observation_shape = (128, 128) # <<< 我帮你把尺寸改成了128x128
        self.reset()

    def reset(self):
        """重置遊戲狀態"""
        self.player = Dinosaur()
        self.cloud = Cloud()
        self.obstacles = []
        self.game_speed = 20
        self.points = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        
        # 繪製初始畫面並返回
        self._draw_elements()
        return self._get_state()

    def step(self, action):
        """
        執行一個動作，讓遊戲前進一幀 (採用先畫再判的邏輯)
        回傳: (state, reward, done)
        """
        # 遊戲事件處理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # === 1. 更新所有遊戲物件的位置 ===
        self.player.update(action)
        
        # 產生新障礙物
        if len(self.obstacles) == 0:
            obst = random.randint(0, 1)
            if obst == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif obst == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                self.obstacles.append(Bird(BIRD))
        
        # 更新障礙物、雲和背景的位置
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
        self.cloud.update(self.game_speed)
        self._update_background()

        # === 2. 繪製更新位置後的新畫面 ===
        # 這一幀的畫面可能就是碰撞發生的畫面
        self._draw_elements()

        # === 3. 從新畫面中獲取狀態 ===
        # 這個 state 是 AI 將要接收到的 next_state
        state = self._get_state()

        # === 4. 進行碰撞檢測和獎勵計算 ===
        reward = 0.1  # 基礎生存獎勵
        done = False

        # 檢查是否成功躲過障礙物
        for obstacle in self.obstacles:
            if not obstacle.passed and self.player.dino_rect.x > obstacle.rect.x + obstacle.rect.width:
                obstacle.passed = True
                reward += 10
                #print("Passed an obstacle! +30 Reward!") # 可以取消註解來除錯

            # 碰撞檢測
            if self.player.dino_rect.colliderect(obstacle.rect):
                done = True
                reward = -100  # 死亡懲罰
                break
        
        # 如果還沒死，才計算得分獎勵
        if not done:
            self.points += 1
            
            if self.points > 0 and self.points % 100 == 0:
                #print(self.points)
                self.game_speed += 1
                reward += 1 # 得分獎勵'''

        # 移除畫面外的障礙物
        self.obstacles = [obs for obs in self.obstacles if obs.rect.x > -obs.rect.width]

        # === 5. 渲染並控制幀率 ===
        if self.render_mode == 'human':
            self.render()
        #self.clock.tick(30) # 建議不要太快，30或60比較穩定

        # === 6. 返回結果 ===
        return state, reward, done

    def _update_background(self):
        """更新背景"""
        image_width = BG.get_width()
        self.x_pos_bg -= self.game_speed
        if self.x_pos_bg <= -image_width:
            self.x_pos_bg = 0

    def _draw_elements(self):
        """在 surface 上繪製所有遊戲元素"""
        self.screen.fill((255, 255, 255))
        
        image_width = BG.get_width()
        self.screen.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))

        self.cloud.draw(self.screen)
        for obstacle in self.obstacles:
            # 障礙物的 draw 方法不再需要 game_speed
            obstacle.draw(self.screen,self.game_speed) 
        self.player.draw(self.screen)

    def _get_state(self):
        """
        從當前畫面擷取狀態 (內容不變)
        """
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.observation_shape[1], self.observation_shape[0]), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, axis=-1)
        return frame

    def render(self):
        """更新螢幕顯示"""
        if self.render_mode == 'human':
            pygame.display.update()

# --- 如何使用 DinoGameEnv 的範例 ---
if __name__ == '__main__':
    # 建立環境，render_mode='human' 才能看到畫面和接收鍵盤輸入
    env = DinoGameEnv(render_mode='human')
    
    episodes = 10
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Episode {episode + 1}/{episodes} ---")
        print("Use UP arrow to JUMP, DOWN arrow to DUCK. Release to RUN.")
        
        while not done:
            # --- 手動控制邏輯 ---
            # 預設動作是 0 (跑步)
            action = 0 
            
            # Pygame 的事件處理，必須放在迴圈內才能即時反應
            # 這是必要的，否則視窗會沒有回應
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # 如果點擊關閉按鈕，則直接結束所有迴圈
                    pygame.quit()
                    quit()

            # 獲取當前所有按鍵的狀態
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]:
                action = 1  # 跳躍
            elif keys[pygame.K_DOWN]:
                action = 2  # 蹲下
            
            # 將選擇的動作傳入環境
            next_state, reward, done = env.step(action)
            print(f"Total reward: {reward:.2f}")
            # 累積獎勵
            total_reward += reward

            # 檢查是否有獲得獎勵（用於除錯）
            if reward != 0.1: # 排除掉基礎生存獎勵
                #print(f"Action: {action}, Reward: {reward:.1f}")
                pass
            
            # 更新狀態（雖然手動玩用不到 state，但保持迴圈完整性）
            state = next_state
            
            # env.step 內部已經包含了 render，所以這裡不需要再呼叫
            # env.render() 
        
        # --- 一局遊戲結束 ---
        print(f"Episode {episode + 1} finished.")
        print(f"Total reward: {total_reward:.2f}")
        
        # 在遊戲結束後等待一會兒，讓玩家看到死亡畫面
        # 並等待任意鍵來開始下一局
        if episode < episodes - 1:
            print("Press any key to start the next episode...")
            waiting_for_key = True
            while waiting_for_key:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        waiting_for_key = False
    
    print("\nAll episodes finished. Exiting.")
    pygame.quit()
    print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
