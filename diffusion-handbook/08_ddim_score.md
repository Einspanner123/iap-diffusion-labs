# 第8章 DDIM 与 Score-Based 模型

> DDPM 的采样需要 1000 步，太慢了。DDIM 发现：去掉随机性，反向过程变成确定性的 ODE，可以大幅跳步加速。同时，Score-Based 模型提供了理解扩散模型的全新视角——不预测噪声，而是学习"概率密度的梯度"。

---

## 8.1 DDIM：确定性采样

### 8.1.1 DDPM 采样的瓶颈

DDPM 的采样公式：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)}_{\text{确定性部分}} + \underbrace{\sigma_t z}_{\text{随机部分}}, \quad z \sim \mathcal{N}(0, I)$$

**问题**：
- 每一步都有随机项 $\sigma_t z$
- 随机性是逐步累积的，跳步会丢失中间的随机性信息
- 必须一步一步走完 1000 步

### 8.1.2 DDIM 的关键洞察

> **DDPM 的 SDE 不是唯一的反向路径！存在一族非马尔可夫的反向过程，它们都共享相同的边缘分布 $q(x_t|x_0)$。**

这意味着：**训练目标不变（因为边缘分布不变），但采样方式可以自由选择。**

### 8.1.3 DDIM 的推导

**核心观察**：DDPM 的训练目标只依赖于边缘分布 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$，不依赖于前向过程是马尔可夫的还是非马尔可夫的。

**DDIM 定义的非马尔可夫前向过程**：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)$$

**验证边缘分布一致性**：给定 $x_0$ 时 $x_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$，由上式采样 $x_{t-1}$，其分布确实是 $\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$。

### 8.1.4 DDIM 采样公式

将 $x_0$ 替换为网络预测 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$，得到 DDIM 采样公式：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}} + \sigma_t z$$

### 8.1.5 η 参数与 DDPM/DDIM 的统一

引入 $\eta \in [0, 1]$ 控制 $\sigma_t$：

$$\sigma_t = \eta \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{\frac{\beta_t}{1-\bar{\alpha}_{t-1}}} = \eta \sqrt{\tilde{\beta}_t}$$

| $\eta$ 值 | 对应方法 | 特点 |
|-----------|---------|------|
| $\eta = 1$ | DDPM | 完全随机，需要逐步采样 |
| $\eta = 0$ | DDIM | 完全确定性，可以跳步 |
| $0 < \eta < 1$ | 介于两者之间 | 部分随机 |

### 8.1.6 DDIM 的跳步采样

当 $\eta = 0$ 时，DDIM 采样变成确定性 ODE，可以自由选择时间步子序列：

$$\tau = \{\tau_1, \tau_2, \ldots, \tau_S\} \subset \{1, 2, \ldots, T\}$$

例如，从 1000 步中选取 50 步：$\tau = \{20, 40, 60, \ldots, 1000\}$

**为什么可以跳步？** 因为确定性 ODE 的轨迹是唯一确定的——从任何中间点出发，沿着 ODE 的方向走，都会到达同一个终点。随机 SDE 则不同——每一步的随机性会影响后续轨迹。

### 8.1.7 DDIM 与 Neural ODE 的关系

DDIM（$\eta = 0$）本质上是对概率流 ODE 的数值求解：

$$\frac{dx}{dt} = \left[-\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)\right]$$

使用欧拉法离散化就得到 DDIM 采样公式。

**Neural ODE 的性质**：
- 可逆性：从 $x_0$ 可以精确恢复 $x_T$
- 精确对数似然：可以通过 ODE 求解器计算
- 连续归一化流：概率守恒

---

## 8.2 Score-Based 模型

### 8.2.1 Score 函数的定义

$$s(x, t) = \nabla_x \log p_t(x)$$

**直觉**：如果 $p_t(x)$ 是一座山的高度图，$s(x, t)$ 就是每个位置的"上山方向"。沿着 Score 方向走，就能走向高概率区域（真实数据）。

### 8.2.2 为什么学习 Score 而不是密度？

直接学习 $p(x)$ 需要计算归一化常数 $Z = \int p_\theta(x) dx$，在高维空间中不可行。

但 Score 函数 $s_\theta(x) = \nabla_x \log p_\theta(x)$ 不需要归一化常数！因为：

$$\nabla_x \log p_\theta(x) = \nabla_x \log \frac{\tilde{p}_\theta(x)}{Z} = \nabla_x \log \tilde{p}_\theta(x) - \underbrace{\nabla_x \log Z}_{= 0}$$

$Z$ 是常数，梯度为 0。所以学习 Score 不需要知道归一化常数。

### 8.2.3 Score Matching

**目标**：训练网络 $s_\theta(x)$ 来估计真实的 Score 函数 $\nabla_x \log p(x)$。

**显式 Score Matching**（Hyvärinen, 2005）：

$$\mathcal{L}_{\text{ESM}} = \mathbb{E}_{p(x)}\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p(x)\|^2\right]$$

问题：需要知道真实的 $\nabla_x \log p(x)$，不可行。

**去噪 Score Matching**（Vincent, 2011）：

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x_0}\mathbb{E}_{q(x_t|x_0)}\left[\frac{1}{2}\|s_\theta(x_t, t) - \nabla_{x_t}\log q(x_t|x_0)\|^2\right]$$

关键：$\nabla_{x_t}\log q(x_t|x_0)$ 是已知的！

对于 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$：

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

**这就是 Score 与噪声预测的关系！** 去噪 Score Matching 等价于噪声预测的 MSE 损失。

### 8.2.4 Langevin 动力学采样

有了 Score 函数后，如何用它来采样？

**Langevin 动力学**：

$$x_{t+1} = x_t + \frac{\alpha}{2} s_\theta(x_t) + \sqrt{\alpha} z, \quad z \sim \mathcal{N}(0, I)$$

其中 $\alpha$ 是步长。

**直觉**：
- $\frac{\alpha}{2} s_\theta(x_t)$：沿着 Score 方向移动（走向高概率区域）
- $\sqrt{\alpha} z$：加入随机扰动（防止陷入局部最优）

当 $\alpha \to 0$ 且步数 $\to \infty$ 时，Langevin 动力学的分布趋向于 $p(x)$。

### 8.2.5 Score-Based 模型的问题与解决

**问题 1：低密度区域 Score 估计不准**

在数据密度低的区域，样本少，Score 估计不准确。Langevin 采样可能在这些区域迷失方向。

**解决**：多尺度噪声扰动——加入不同强度的噪声，在噪声"填满"低密度区域后估计 Score。这就是 NCSN（Noise Conditional Score Network）的核心思想。

**问题 2：单一噪声水平不够**

只用一种噪声水平，要么高噪声破坏数据结构，要么低噪声无法覆盖低密度区域。

**解决**：使用一系列递增的噪声水平 $\sigma_1 < \sigma_2 < \cdots < \sigma_L$，先在高噪声下采样（快速接近数据流形），再逐步降低噪声（精细化）。

---

## 8.3 DDPM、DDIM、Score-Based 的统一视角

| 视角 | 方法 | 核心思想 |
|------|------|---------|
| 概率模型 | DDPM | 学习反向过程的条件分布 |
| 确定性变换 | DDIM | 用 ODE 描述反向轨迹 |
| Score 函数 | Score-Based | 学习概率密度的梯度 |

**三者等价的核心**：

$$\epsilon_\theta(x_t, t) \quad \longleftrightarrow \quad s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}} \quad \longleftrightarrow \quad \text{ODE 的向量场}$$

预测噪声 = 预测 Score = 定义 ODE 的方向

---

## 8.4 本章小结

| 方法 | 核心贡献 | 采样方式 | 采样速度 |
|------|---------|---------|---------|
| DDPM | 建立扩散模型框架 | 随机 SDE | 1000 步 |
| DDIM | 确定性 ODE 采样 | 确定 ODE | 20-100 步 |
| Score-Based | Score 函数视角 | Langevin 动力学 | 取决于步数 |

**关键数学工具**：
- DDIM：非马尔可夫过程、ODE 离散化、边缘分布一致性
- Score-Based：梯度、Score Matching、Langevin 动力学

---

## 练习与思考

1. **DDIM 采样**：设 $\eta = 0$，$\bar{\alpha}_{t-1} = 0.9$，$\bar{\alpha}_t = 0.85$，$x_t = 0.5$，$\epsilon_\theta(x_t, t) = 0.3$。计算 DDIM 采样的 $x_{t-1}$。

2. **Score 计算**：设 $q(x_t|x_0) = \mathcal{N}(\sqrt{0.8}x_0, 0.2I)$。计算 $\nabla_{x_t}\log q(x_t|x_0)$，并验证它等于 $-\epsilon/\sqrt{0.2}$。

3. **Langevin 动力学**：设 $s_\theta(x) = -x$（标准高斯分布的 Score），$\alpha = 0.1$，$x_0 = 2$。手动执行 3 步 Langevin 动力学（忽略随机项），观察 $x$ 的变化。

4. **思考题**：DDIM 为什么可以复用 DDPM 训练的权重？从"边缘分布一致性"的角度解释。
