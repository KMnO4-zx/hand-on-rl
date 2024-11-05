# hand-on-rl

我们以超级马里奥游戏为例，逐步了解强化学习中涉及的基本概念：

- 环境 (Environment)：环境是智能体所处的外部系统，它负责产生当前的状态，接收智能体的动作并返回新的状态和对应的奖励。环境的作用相当于模拟现实中的条件和反应规则，智能体只能通过与环境的交互来了解其动态变化。以超级马里奥游戏为例，环境包括玩家看到的游戏画面和后台程序逻辑。环境控制了游戏进程，例如生成敌人、提供奖励以及决定游戏何时结束。智能体并不知道环境的内部实现细节，只能依靠输入输出规则与环境互动。
- 智能体 (Agent)：智能体是强化学习中的决策者，它会不断地观察环境的状态，并根据其策略选择动作。智能体的目标是通过选择一系列最优动作，获得尽可能多的累积奖励。
- 状态 (State)：状态是环境在特定时刻的全面描述。对于智能体而言，状态是决策的基础，它包含了关于当前环境的所有重要信息。
- 动作 (Action)：动作是智能体对当前状态的反应。基于当前的状态，智能体使用其策略函数来决定下一步要采取的动作。例如，在超级玛丽中，动作可以包括“向左移动”、“向右移动”和“跳跃”。动作可以是离散的（如跳跃或移动的方向选择）或者连续的（如机器手臂在三维空间中的移动角度）。强化学习的核心在于使智能体学会如何在每个状态下选择最优的动作，从而最大化回报。
- 奖励 (Reward)：奖励是环境对智能体执行动作后给予的反馈。奖励可以是正的（奖励）或者负的（惩罚）。例如，在超级马里奥游戏中，吃到金币可以获得正奖励（例如 +10 分），而碰到敌人会得到负奖励（例如 -100 分）。
- 动作空间 (Action Space)：指智能体在当前状态下可以选择的动作集合。
- 轨迹 (Trajectory)：轨迹（又称为回合或episode）是指智能体在一次完整的交互过程中经历的一系列状态、动作和奖励的序列。轨迹通常表示为 $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$，其中 $s_i$ 表示第 $i$ 时刻的状态，$a_i$ 表示智能体在状态 $s_i$ 下选择的动作。。比如大语言模型生成时，它的状态就是已经生成的token序列。当前的动作是生成下一个token。当前token生成后，已生成的序列就加上新生成的token成为下一个状态。
- 回报 (Return Reward)：表示从当前时间步开始直到未来的累积奖励和，通常用符号 $G_t$ 表示：$G_t = R_{t+1} + R_{t+2} + \dots + R_T$。回报的定义是智能体决策的重要依据，因为强化学习的目标是训练一个策略，使得智能体在每个状态下的期望回报最大化。

<p align="center">
    <img src="./images/image-1.png" alt="alt text" width="60%">
</p>

## 强化学习的目标

在强化学习中，目标是训练一个神经网络 $Policy$ $\pi$ ，在所有状态 $s$ 下，给出相应的 $Action$ ，得到的 $Return$ 的期望值最大。即：
$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

其中：
1. $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$：表示在策略 $P_{\theta}(\tau)$ 下轨迹 $\tau$ 的回报 $R(\tau)$ 的期望值。
2. $R(\tau)$：轨迹 $\tau$ 的回报，即从起始状态到终止状态获得的所有奖励的总和。
3. $\tau$：表示一条轨迹，即智能体在环境中的状态和动作序列。
4. $P_{\theta}(\tau)$：在参数 $\theta$ 下生成轨迹 $\tau$ 的概率，通常由策略或策略网络确定。
5. $\theta$：策略的参数，控制着策略 $P_{\theta}$ 的行为。

所以，我们的目标是找到一个策略 $\pi$，使得 $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ 最大。那怎么找到这个策略呢？我们使用梯度上升的办法，即不断地更新策略参数 $\theta$，使得 $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ 不断增大。

首先，我们来计算梯度：

$$
\begin{align*}
\nabla E(R(\tau))_{\tau \sim P_{\theta}(\tau)} &= \nabla \sum_{\tau} R(\tau) P_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) \nabla P_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) \nabla P_{\theta}(\tau) \frac{P_{\theta}(\tau)}{P_{\theta}(\tau)} \\
&= \sum_{\tau} P_{\theta}(\tau) R(\tau) \frac{\nabla P_{\theta}(\tau)}{P_{\theta}(\tau)} \\
&= \sum_{\tau} P_{\theta}(\tau) R(\tau) \nabla \log P_{\theta}(\tau) \\
&\approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log P_{\theta}(\tau^n)
\end{align*}
$$

接下来，我们来看一下 Trajectory 的概率 $P_{\theta}(\tau)$ 是怎么计算的：

$$
\begin{align*}
\frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log P_{\theta}(\tau^n) &= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log \prod_{t=1}^{T_n} P_{\theta}(a_n^t \mid s_n^t) \\
&= \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \sum_{t=1}^{T_n} \nabla \log P_{\theta}(a_n^t \mid s_n^t) \\
&= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log P_{\theta}(a_n^t \mid s_n^t) \\
&= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t \mid s_n^t)
\end{align*}
$$

1. $\frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log P_{\theta}(\tau^n)$：对轨迹$\tau^n$ 的概率对数进行求导，表示利用策略梯度对期望回报进行优化。

2. $\prod_{t=1}^{T_n} P_{\theta}(a_n^t | s_n^t)$：表示轨迹$\tau^n$ 中所有步骤$t$上采取的动作$a_n^t$ 在状态$s_n^t$ 下的联合概率。

3. $\nabla \log \prod_{t=1}^{T_n} P_{\theta}(a_n^t | s_n^t)$：利用对数的可加性，将联合概率的对数梯度分解为各步的对数梯度之和。

4. $\sum_{t=1}^{T_n} \nabla \log P_{\theta}(a_n^t | s_n^t)$：对每一步动作的概率对数取梯度，分解为每一步的累加。

5. $\sum_{t=1}^{T_n} R(\tau^n) \nabla \log P_{\theta}(a_n^t | s_n^t)$：利用累积回报$R(\tau^n)$ 加权每一步的对数梯度，体现策略梯度方法中的优势估计。

6. $\sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t | s_n^t)$：省略梯度符号后的形式，通常用于描述带有加权对数概率的情况。

那我们应该如何训练一个 Policy 网络呢？受局限我们可以定义loss函数为：

$$
loss = - \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t \mid s_n^t)
$$

在我们的目标函数前加上负号，就可以转化为一个最小化问题。我们可以使用梯度下降的方法来求解这个问题。
但是，我们在实际训练中，通常会使用更加稳定的方法，即使用基于策略梯度的方法，例如 PPO、TRPO 等。

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_{\theta}(a_n^t \mid s_n^t)
$$

如以上公式所示，如果当前的 Trajectory 的回报 $R(\tau)$ 较大，那么我们就会增大这个 Trajectory 下所有 $action$ 的概率，反之亦然。
这样，我们就可以不断地调整策略，使得回报最大化。
但这明显是存在改进空间的，因为我们只是简单地使用回报来调整策略，而没有考虑到回报的分布，这样就会导致回报的方差较大，训练不稳定。

针对这个问题，我们修改一下公式，首先对 $Reward$ 求和：

$$
R(\tau^n) \rightarrow \sum_{t' = t}^{T_n} \gamma^{t' - t} r_{t'}^n = R_t^n
$$

其中：
1. $R(\tau^n)$：轨迹 $\tau^n$ 的累积回报，这里使用了未来回报的折扣求和来表示。

2. $R_t^n$：从时间步$t$开始的未来折扣回报，表示轨迹$\tau^n$在时间步$t$时的累计回报。

3. $\sum_{t' = t}^{T_n}$：对时间步$t$到$T_n$（轨迹结束时刻）之间的所有奖励进行求和。

4. $\gamma^{t' - t}$：折扣因子$\gamma$的幂次，控制未来奖励的权重，$\gamma \in [0,1]$。当$t'$ 越远离当前时刻 $t$，其贡献越小。

5. $r_{t'}^n$：在时间步 $t'$发生的即时奖励。

总的来说，修改后的公式是对未来回报的折扣求和，这样当前动作的概率就不再只取决于当前的回报，而是取决于未来的回报，这样就可以减小回报的方差，使得训练更加稳定。

还有一种情况会影响我们算法的稳定性，那就是在好的局势下和坏的局势下。比如在好的局势下，不论你做什么动作，你都会得到正的回报，这样算法就会增加所有动作的概率。
得到reward大的动作的概率大一些，但是这样会让训练很慢，也会不稳定。最好是能够让相对好的动作的概率增加，相对坏的动作的概率减小。

为了解决这个问题，我们可以对所有动作的reward都减去一个baseline，这样就可以让相对好的动作的reward增加，相对坏的动作的reward减小，也能反映这个动作相对其他动作的价值。

所以我们的目标函数就变为：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n - B(s_n^t)) \nabla \log P_{\theta}(a_n^t \mid s_n^t)
$$

其中，$B(s_n^t)$ 也需要用神经网络来拟合，这就是我们的 Actor-Critic 网络。Actor网络负责输出动作的概率，Critic网络负责评估Actor网络输出的动作好坏。

接下来我们再来解释几个常见的强化学习概念：

- Action-Value Function：$R_t^n$每次都是随机采样，方差很大，我们可以用 $Q_{\theta}(s, a)$ 来代替，$Q_{\theta}(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 的价值，即从状态 $s$ 开始，采取动作 $a$ 后，按照某个策略 $\pi$ 执行，最终获得的回报的期望值。$Q(s, a)$ 可以用来评估在状态 $s$ 下采取动作 $a$ 的好坏，从而指导智能体的决策，即动作价值函数。

- State-Value Function：$V_{\theta}(s)$ 表示在状态 $s$ 下的价值，即从状态 $s$ 开始，按照某个策略 $\pi$ 执行，最终获得的回报的期望值。$V(s)$ 可以用来评估在状态 $s$ 下的好坏，从而指导智能体的决策，即状态价值函数。

- Advantage Function：$A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)$，表示在状态 $s$ 下采取动作 $a$ 相对于采取期望动作的优势。优势函数可以用来评估在状态 $s$ 下采取动作 $a$ 的优劣，从而指导智能体的决策，即优势函数。

有了这些概念，我们再回过头来看我们的目标函数：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n - B(s_n^t)) \nabla \log P_{\theta}(a_n^t \mid s_n^t)
$$

其中：$R_t^n - B(s_n^t)$就是我们刚刚讲的优势函数，表示在状态 $s_n^t$ 下采取动作 $a_n^t$ 相对于采取期望动作的优势。那我们的目标函数就变成了：最大化优势函数的期望。

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}(s_n^t, a_n^t) \nabla \log P_{\theta}(a_n^t \mid s_n^t)
$$

那如何计算优势函数呢？我们重新来看一下优势函数的定义：

$$
A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)
$$

$Q_{\theta}(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 的价值，$V_{\theta}(s)$ 表示在状态 $s$ 下的价值。我们来看一下下面这个公式：

$$
Q_\theta(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1})
$$

其中：
1. $r_t$：执行动作 $a$ 后，在状态 $s_t$ 下获得的即时奖励。

2. $\gamma$：折扣因子，用于确定未来奖励的重要性。折扣因子接近1时，更加关注未来的奖励；接近0时，更加重视即时奖励。

3. $V_\theta(s_{t+1})$：价值函数，用参数$\theta$ 表示，估计下一个状态 $s_{t+1}$ 的价值，即从该状态开始的预期未来奖励。


我们把上述公式代入到优势函数的定义中：

$$
\begin{align*}
A_{\theta}(s_t, a) &= Q_{\theta}(s_t, a) - V_{\theta}(s_t) \\ 
&= r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
\end{align*}
$$

我们可以看到，现在优势函数中只剩下了状态价值函数 $V_\theta(s_t)$ 和下一个状态的价值函数 $V_\theta(s_{t+1})$，这样就由原来需要训练两个神经网络变成了只需要训练一个状态价值网络，这样就减少了训练的复杂度。

在上面的函数中，我们是对Reward进行一步采样，下面我们对状态价值函数也进行action和reward的一步采样。

$$
V_\theta(s_{t+1}) \approx r_{t+1} + \gamma \cdot V_\theta(s_{t+2})
$$

接下里，我们就可以对优势函数进行多步采样，也可以全部采样。

从图片中提取的公式为：

$$
A_\theta^1(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
A_\theta^2(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot V_\theta(s_{t+2}) - V_\theta(s_t)
$$

$$
A_\theta^3(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \gamma^3 \cdot V_\theta(s_{t+3}) - V_\theta(s_t)
$$

$$
\vdots
$$

$$
A_\theta^T(s_t, a) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \gamma^3 \cdot r_{t+3} + \cdots + \gamma^T \cdot r_T - V_\theta(s_t)
$$

我们知道，采样的步数越多，会导致方差越大，但偏差会越小。为了让式子更加简洁，定义：

$$
\delta_t^V = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
\delta_{t+1}^V = r_{t+1} + \gamma \cdot V_\theta(s_{t+2}) - V_\theta(s_{t+1})
$$

其中：

1. $\delta_t^V$：是时间步$t$的优势函数，表示当前时刻$t$的即时奖励 $r_t$ 加上下一个状态的折扣价值$\gamma \cdot V_\theta(s_{t+1})$ 减去当前状态的估计价值 $V_\theta(s_t)$。

2. $\delta_{t+1}^V$：是时间步 $t+1$ 的优势函数，类似地表示在时刻 $ t+1 $ 获得的即时奖励 $ r_{t+1}$ 加上状态 $ s_{t+2} $ 的折扣价值 $ \gamma \cdot V_\theta(s_{t+2}) $减去状态 $ s_{t+1} $ 的价值估计 $ V_\theta(s_{t+1}) $。

那我们究竟要采样几步呢？介绍一下广义优势估计GAE（Generalized Advantage Estimation），小孩子才做选择，我（GAE）全都要。

$$
A_\theta^{\text{GAE}}(s_t, a) = (1 - \lambda) (A_\theta^1(s_t, a) + \lambda A_\theta^2(s_t, a) + \lambda^2 A_\theta^3(s_t, a) + \cdots)
$$

假如$\lambda = 0.9$:

$$
A_\theta^{\text{GAE}}(s_t, a) = 0.1 \cdot A_\theta^1(s_t, a) + 0.9 \cdot A_\theta^2(s_t, a) + 0.9^2 \cdot A_\theta^3(s_t, a) + \cdots
$$

将上面定义好的$\delta_t^V$和$\delta_{t+1}^V$代入到GAE优势函数中：

$$
A_\theta^{\text{GAE}}(s_t, a) = (1 - \lambda) (\delta_t^V + \lambda \cdot \delta_{t+1}^V + \lambda^2 \cdot \delta_{t+2}^V + \cdots)
$$

最终我们可以得到：

$$
A_\theta^{\text{GAE}}(s_t, a) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

## Proximal Policy Optimization (PPO) 邻近策略优化

PPO 是 OpenAI 提出的一种基于策略梯度的强化学习算法，它通过对策略梯度的优化，来提高策略的稳定性和收敛速度。PPO 算法的核心思想是在更新策略时，通过引入一个重要性采样比例，来限制策略更新的幅度，从而保证策略的稳定性。

PPO 算法的目标函数为：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
$$

其中：
1. $\frac{1}{N} \sum_{n=1}^{N}$：对 $N$ 条轨迹（采样的样本）取平均值。这里的 $N$ 表示采样轨迹的总数，通过对多个轨迹求平均来估计梯度，以获得更稳定的更新。

2. $\sum_{t=1}^{T_n}$：对每条轨迹 $n$ 中的 $T_n$ 个时间步求和，表示对单条轨迹中的所有时间步的累积。

3. $A_{\theta'}^{GAE}(s_n^t, a_n^t)$：广义优势估计（Generalized Advantage Estimation, GAE），由参数 $\theta'$ 估计，用于计算在状态 $s_n^t$ 下采取动作 $a_n^t$ 的优势。

4. $\frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}$：表示策略的梯度，其中分母 $P_{\theta'}(a_n^t | s_n^t)$ 是旧策略（或目标策略），分子 $\nabla P_\theta(a_n^t | s_n^t)$ 是新策略的梯度。这个比值反映了新旧策略在同一状态-动作对上的相对概率密度，利用这一比值来更新策略参数 $\theta$。

整个公式的作用是通过优势估计来计算策略梯度，以优化策略参数，使得策略倾向于选择优势更高的动作，从而提升策略性能。GAE 可以有效降低方差，使得策略优化过程更加稳定和高效。

还是将loss函数取负号，转化为最小化问题，我们可以得到：

$$
loss = - \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}
$$

具体来说，PPO 算法主要包括两个关键的技术：Adaptive KL Penalty Coefficient 和 Clipped Surrogate Objective。

PPO-惩罚（PPO-Penalty）用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，这就变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。即：

$$
Loss_{kl} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)} + \beta KL(P_\theta, P_{\theta'})
$$

其中：

- $\beta KL(P_\theta, P_{\theta'})$：这是KL散度项，用于限制新旧策略之间的距离，其中 $KL(P_\theta, P_{\theta'})$ 表示策略$P_\theta$和旧策略$P_{\theta'}$之间的KL散度。超参数$\beta$控制KL散度项的权重，从而调节新旧策略之间的差异，防止策略更新过大导致不稳定。

整个PPO-KL损失函数的目的是通过限制新旧策略的差异（使用KL散度项）来优化策略，使其更稳定地朝着优势更高的方向进行更新。PPO的这种约束策略更新的方法相比于其他策略优化方法更为稳定且有效。

PPO截断（PPO-Clipped）是 PPO 的另一种变体，它通过对比新旧策略的比值，来限制策略更新的幅度，从而保证策略的稳定性。具体来说，PPO-Clipped 的目标函数为：

$$
Loss_{clip} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \min \left( A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, \, \text{clip} \left( \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, 1 - \epsilon, 1 + \epsilon \right) A_{\theta'}^{GAE}(s_n^t, a_n^t) \right)
$$

- $\text{clip} \left( \frac{P_\theta(a_n^t | s_n^t)}{P_{\theta'}(a_n^t | s_n^t)}, 1 - \epsilon, 1 + \epsilon \right)$：裁剪函数，将概率比裁剪到 $[1 - \epsilon, 1 + \epsilon]$ 区间，防止策略的更新步长过大。这里$\epsilon$ 是一个超参数，控制裁剪的范围。

- $\min(\cdot, \cdot)$：在未裁剪的概率比项和裁剪后的项之间取最小值。这一操作的目的在于限制策略更新幅度，以防止策略偏离旧策略过远，从而导致不稳定的学习过程。

整个PPO-clip损失函数的作用是通过裁剪操作约束策略的变化幅度，使策略更新不会过于激进。这种方式相比于传统策略梯度方法更为稳定，并且在优化过程中能够有效平衡探索和利用。PPO2 的这种裁剪机制是其成功的关键，广泛用于实际的强化学习应用中。

好了，如果你坚持看到了这里，那想必你已经差不多掌握了强化学习的基本思想和PPO算法的基本思想。接下来你可以将PPO应用到大模型的训练中啦！

**参考文献**

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [动手学强化学习](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/)
3. [零基础学习强化学习算法：ppo](https://www.bilibili.com/video/BV1iz421h7gb/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=c102de6ffc75a54d6576f9fdc931e08a)