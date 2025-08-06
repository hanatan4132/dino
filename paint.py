import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_experiment_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['mean']), np.array(data['std'])

def plot_single_result(mean, std, label):
    episodes = np.arange(len(mean))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, mean, label=label)
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Performance of {label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_results_together(json_paths, title="All Experiments Comparison"):
    plt.figure(figsize=(10, 5))
    for path in json_paths:
        mean, std = load_experiment_results(path)
        episodes = np.arange(len(mean))
        label = os.path.splitext(os.path.basename(path))[0]
        plt.plot(episodes, mean, label=label)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 設定資料夾路徑與檔案路徑列表
json_dir = "experiment_results"
json_paths = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

# 逐一畫出每個結果
for path in json_paths:
    mean, std = load_experiment_results(path)
    label = os.path.splitext(os.path.basename(path))[0]
    plot_single_result(mean, std, label)

# 疊圖比較
plot_all_results_together(json_paths)

'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def load_experiment_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['mean']), np.array(data['std'])

def plot_group_results(json_paths, title):
    plt.figure(figsize=(10, 5))
    for path in json_paths:
        mean, std = load_experiment_results(path)
        episodes = np.arange(len(mean))
        label = os.path.splitext(os.path.basename(path))[0]
        plt.plot(episodes, mean, label=label)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 取得所有 json 路徑
json_dir = "experiment_results"
all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

# 分出 target 與 others
target_path = os.path.join(json_dir, "target.json")
other_paths = [p for p in all_jsons if os.path.basename(p) != "target.json"]

# 組合每組包含 target 的三個組合
group_count = 0
for combo in combinations(other_paths, 2):  # 從其他檔案中取兩個
    group_paths = [target_path] + list(combo)
    group_title = f"Group {group_count + 1}: " + ", ".join([os.path.splitext(os.path.basename(p))[0] for p in group_paths])
    plot_group_results(group_paths, group_title)
    group_count += 1
'''