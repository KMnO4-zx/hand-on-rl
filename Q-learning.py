import numpy as np
import random

# 定义一个简单的环境（简单的网格世界）
class SimpleEnv:
    def __init__(self):
        self.state_space = 5  # 定义状态空间大小为5
        self.action_space = 2  # 定义动作空间大小为2：0（左移），1（右移）
        self.state = 0  # 初始状态为0

    def reset(self):
        self.state = 0  # 重置环境到初始状态
        return self.state

    def step(self, action):
        # 根据所选动作定义下一状态及奖励
        if action == 0:  # 动作为0：向左移动
            next_state = max(0, self.state - 1)  # 防止状态小于0
        else:  # 动作为1：向右移动
            next_state = min(self.state_space - 1, self.state + 1)  # 防止状态超过边界

        # 定义奖励逻辑
        reward = 1 if next_state == self.state_space - 1 else 0  # 到达最右侧边界给予奖励1
        done = next_state == self.state_space - 1  # 判断是否达到终止状态

        self.state = next_state
        return next_state, reward, done

# Q-Learning算法的核心实现
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    # 初始化Q表，所有Q值初始为0
    Q = np.zeros((env.state_space, env.action_space))

    # 迭代多次以模拟多个训练回合
    for episode in range(episodes):
        state = env.reset()  # 每个回合开始时重置环境
        done = False  # 标记回合是否结束

        # 持续执行动作直到回合结束
        while not done:
            # 采用ε-greedy策略选择动作
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # 探索：以ε的概率随机选择动作
            else:
                action = np.argmax(Q[state, :])  # 利用：以1-ε的概率选择当前最优动作

            # 执行动作并获取下一状态、奖励和是否结束的标记
            next_state, reward, done = env.step(action)

            # 使用Q-Learning公式更新Q值
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )

            state = next_state  # 更新当前状态

    return Q

# 运行Q-Learning算法
env = SimpleEnv()
Q_table = q_learning(env)
print("学习到的Q表:")
print(Q_table)