# 第三代：Score SDE (Score-based Generative Models via SDE)

> **论文：** Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., ICLR 2021)
>
> **地位：** 将 DDPM 和 Score-based 模型统一到 SDE 框架下，揭示了扩散模型与 ODE/SDE 的深层联系

---

## 一、核心思想

### 1.1 从离散到连续

```
DDPM / DDIM:                    Score SDE:
┌──────────────────────┐        ┌──────────────────────────┐
│ 离散时间步 t = 0,1,...,T │        │ 连续时间 t ∈ [0, 1]       │
│                      │        │                          │
│ q(x_t|x_{t-1})       │   统一  │ dx = f(x,t)dt + g(t)dW    │
│ = N(√(1-β)x, βI)     │  ──→   │                          │
│                      │        │ f(x,t): 漂移项            │
│ 需要设计 β₁,...,β_T   │        │ g(t): 扩散系数            │
└──────────────────────┘        └──────────────────────────┘
 离散的噪声调度                    连续的微分方程
```

### 1.2 什么是 Score 函数？

$$\boxed{s(x, t) := \nabla_x \log p_t(x)}$$

| 符号 | 含义 |
|------|------|
| $\nabla_x$ | 对 $x$ 的梯度（向量） |
| $p_t(x)$ | 时间 $t$ 处的概率密度 |
| $s(x, t)$ | **Score 函数**——指向概率密度增长最快的方向 |

**直觉：**
- 如果 $p_t(x)$ 是一座山的高度图
- $s(x, t) = \nabla_x \log p_t(x)$ 就是每个位置的"上山方向"
- **沿着 score 方向走 → 走向高概率区域（真实数据）**

---

## 二、从 DDPM 到 SDE 的推导

### 2.1 回顾 DDPM 的前向过程

DDPM 前向过程（离散）：

$$x_{t} = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

重写为增量形式：

$$x_{t} - x_{t-1} = (\sqrt{1-\beta_t} - 1) x_{t-1} + \sqrt{\beta_t} \epsilon$$

令 $\Delta t = 1/T$（每步的时间间隔），则：

$$\frac{x_{t} - x_{t-1}}{\Delta t} = T(\sqrt{1-\beta_t} - 1) x_{t-1} + T\sqrt{\beta_t} \epsilon$$

### 2.2 连续化极限

当 $T \to \infty$（步数趋近无穷），$\beta_t \to 0$，利用泰勒展开：

$$\sqrt{1-\beta_t} \approx 1 - \frac{\beta_t}{2}, \quad \sqrt{\beta_t} \approx \sqrt{\beta_t}$$

定义连续时间的噪声调度 $\beta(t) = \lim_{T\to\infty} \beta_{\lfloor tT \rfloor}$

得到：

$$\mathrm{d}x = \underbrace{-\frac{1}{2}\beta(t)x}_{f(x,t)}\,\mathrm{d}t + \underbrace{\sqrt{\beta(t)}}_{g(t)}\,\mathrm{d}W_t$$

这就是 **VP-SDE (Variance Preserving SDE)**！

### 2.3 两种标准 SDE 形式

#### VP-SDE (Variance Preserving)

$$\mathrm{d}x = -\frac{1}{2}\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t$$

| 性质 | 说明 |
|------|------|
| 来源 | DDPM 的连续极限 |
| 特点 | 保持方差结构 |
| 边缘分布 | $p_{0t}(x|x_0) = \mathcal{N}(x; e^{-\frac{1}{2}\int_0^t \beta(s)ds}x_0, (1-e^{-\int_0^t \beta(s)ds})I)$ |

#### VE-SDE (Variance Exploding)

$$\mathrm{d}x = g(t)^2\nabla_x \log p_t(x)\,\mathrm{d}t + g(t)^2\,\mathrm{d}t\,\mathrm{d}W_t$$

| 性质 | 说明 |
|------|------|
| 来源 | Score-based 模型 |
| 特点 | 方差随时间爆炸式增长 |
| 边缘分布 | $p_{0t}(x|x_0) = \mathcal{N}(x; x_0, (\int_0^t g^2(s)ds)I)$ |

**两种 SDE 通过变量替换可以互相转换！**

---

## 三、Score Matching 训练方法

### 3.1 为什么学 Score？

**核心观察：** 给定 SDE 的漂移项和扩散项，如果我们知道 score 函数 $\nabla_x \log p_t(x)$，就可以构造反向 SDE 进行采样。

**反向 SDE：**

$$\mathrm{d}x = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,\mathrm{d}t + g(t)\,\overline{\mathrm{d}W}_t$$

其中 $\overline{\mathrm{d}W}_t$ 是反向布朗运动。

> 🎯 **关键：** 反向 SDE 中需要知道 $\nabla_x \log p_t(x)$ —— 这就是神经网络要学习的东西！

### 3.2 Denoising Score Matching

直接估计 $\nabla_x \log p_t(x)$ 很困难。**Denoising Score Matching** 提供了一个巧妙的方法：

**定理（Denoising Score Matching）：**

$$\mathbb{E}_{p_t(x)}[\|\nabla_x \log p_t(x) - s_\theta(x,t)\|^2] = \mathbb{E}_{q(x_0), t, \tilde{x}}[\|\nabla_{\tilde{x}} \log q(\tilde{x}|x_t) - s_\theta(\tilde{x},t)\|^2] + C$$

其中：
- 左边：我们想优化的目标（score 匹配）
- 右边：我们可以计算的目标（去噪匹配）
- $q(\tilde{x}|x_t)$ 是以 $x_t$ 为中心的高斯扰动分布
- $C$ 是常数

**直觉：** 学习从带噪数据中恢复干净数据的梯度，等价于学习数据的 score。

### 3.3 具体损失函数

对于 VE-SDE，使用**加权损失函数**：

$$\boxed{\mathcal{L}(\theta) = \frac{1}{2}\mathbb{E}_{t}\left[ \lambda(t) \mathbb{E}_{x(0), x(t)}\left[\|s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t)|x(0))\|^2\right] \right]}$$

对于 VP-SDE（即 DDPM 的连续形式）：

$$\nabla_{x(t)} \log p_{0t}(x(t)|x(0)) = -\frac{x(t) - e^{-\frac{1}{2}\int_0^t \beta(s)ds}x(0)}{1 - e^{-\int_0^t \beta(s)ds}} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}(t)}}$$

这恰好就是 DDPM 的噪声预测目标！（相差一个常数因子）

---

## 四、采样方法

### 4.1 随机采样器（SDE Solver）

**算法：Euler-Maruyama 方法求解反向 SDE**

| 步骤 | 操作 |
|------|------|
| **输入** | Score 网络 $s_\theta$，SDE 参数 $f, g$，步数 $N$ |
| **初始化** | $x_N \sim p_{\text{prior}}$（通常是 $\mathcal{N}(0, I)$），$h = T/N$ |
| **循环** | 对 $i = N-1, N-2, \ldots, 0$： |
| ① | 计算 drift: $\hat{f} = f(x_i, t_i) - g(t_i)^2 s_\theta(x_i, t_i)$ |
| ② | 采样 $\epsilon \sim \mathcal{N}(0, I)$ |
| ③ | $x_{i-1} = x_i + h \cdot \hat{f} + g(t_i)\sqrt{h} \cdot \epsilon$ |
| **输出** | $x_0$ |

### 4.2 确定性采样器（ODE Solver / Probability Flow ODE）

**核心发现：** 每个 SDE 都对应一个**概率流 ODE**，其边缘分布与 SDE 完全相同！

**概率流 ODE：**

$$\mathrm{d}x = \underbrace{[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]}_{\text{确定性漂移}}\,\mathrm{d}t$$

**算法：Euler 方法求解概率流 ODE**

| 步骤 | 操作 |
|------|------|
| **输入** | Score 网络 $s_\theta$，SDE 参数 $f, g$，步数 $N$ |
| **初始化** | $x_N \sim p_{\text{prior}}$，$h = T/N$ |
| **循环** | 对 $i = N-1, N-2, \ldots, 0$： |
| ① | 计算 drift: $\hat{f} = f(x_i, t_i) - \frac{1}{2}g(t_i)^2 s_\theta(x_i, t_i)$ |
| ② | $x_{i-1} = x_i + h \cdot \hat{f}$ |
| **输出** | $x_0$ |

> ⚠️ 注意与随机采样的区别：**没有 $g(t)\sqrt{h}\cdot\epsilon$ 项！**

### 4.3 预测校正采样器（Predictor-Corrector）

结合两者优势的高级采样器：

```
for each step:
    Predictor: 用 ODE 大步前进（快速）
    Corrector: 用 SDE 小步修正（精确）
```

---

## 五、与前两代的详细对比

### 5.1 公式对比总表

| 项目 | DDPM (2020) | DDIM (2021) | Score SDE (2021) |
|------|-------------|-------------|------------------|
| **数学框架** | 离散马尔可夫链 | 离散非马尔可夫 | **连续 SDE/ODE** |
| **前向过程** | $q(x_t\|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | 同左 | $\mathrm{d}x = f(x,t)\mathrm{d}t + g(t)\mathrm{d}W_t$ |
| **学习目标** | 噪声预测 $\epsilon_\theta$ | 噪声预测 $\epsilon_\theta$ | **Score 预测 $s_\theta = \nabla_x \log p_t(x)$** |
| **损失函数** | $\|\epsilon - \epsilon_\theta\|^2$ | 同左 | $\lambda(t)\|s_\theta - \nabla \log p_{0t}\|^2$ |
| **噪声调度** | $\beta_t$（超参数） | 复用 DDPM 的 | $g(t)$ 或 $\beta(t)$（仍需设计） |
| **采样方式** | 随机（1000步） | 确定/可选（50步） | 随机或确定（可变步数） |
| **理论统一性** | 单一模型 | DDPM 变体 | **统一所有扩散模型** |

### 5.2 关键参数对比

| 参数 | DDPM | DDIM | Score SDE |
|------|------|------|-----------|
| 时间表示 | 离散 $t \in \{1,\ldots,T\}$ | 同左 | **连续 $t \in [0,T]$** |
| 噪声调度 | $\beta_t \in (0,1)$ | 同左 | $g(t): [0,T] \to \mathbb{R}^+$ |
| 网络输出 | $\hat{\epsilon} \in \mathbb{R}^d$ | 同左 | **$\hat{s} \in \mathbb{R}^d$（score 向量）** |
| 控制参数 | 无 | $\eta \in [0,1]$ | $\lambda(t)$（加权函数） |
| 扩散系数 | $\sigma_t$（固定或 $\tilde{\beta}_t$） | 被 $\eta$ 取代 | **$g(t)$（连续函数）** |

### 5.3 Score 与 Noise 的关系

在 VP-SDE（DDPM 的连续版本）中：

$$s_\theta(x,t) = -\frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar{\alpha}(t)}}$$

| 关系 | 说明 |
|------|------|
| **Score = -Noise / 缩放因子** | 两者只差一个常数缩放 |
| **Score 更通用** | 可以处理任意 SDE，不限于 VP 类型 |
| **Noise 更直观** | "预测加了多少噪声"更容易理解 |

---

## 六、Score SDE 的完整变量表

| 变量 | 类型 | 定义域 | 说明 |
|------|------|--------|------|
| $x(t)$ | 数据向量 | $\mathbb{R}^d$ | 连续时间 $t$ 时的数据状态 |
| $t$ | 连续时间 | $[0, T]$ | 当前时刻（通常 $T=1$） |
| $f(x,t)$ | 漂移场 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | SDE 的确定性部分 |
| $g(t)$ | 扩散系数 | $[0,T] \to \mathbb{R}^+$ | 控制 SDE 随机性强度 |
| $W_t$ | 布朗运动 | — | 标准维纳过程 |
| $s(x,t)$ | Score 函数 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | $\nabla_x \log p_t(x)$ |
| $s_\theta(x,t)$ | 神经网络 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | Score 的近似 |
| $\lambda(t)$ | 权重函数 | $[0,T] \to \mathbb{R}^+$ | 不同时间步的损失权重 |
| $p_t(x)$ | 概率密度 | $\mathbb{R}^d \to \mathbb{R}^+$ | 时间 $t$ 时的数据分布 |
| $p_{0t}(x\|x_0)$ | 条件密度 | — | 给定 $x_0$ 时 $x(t)$ 的分布 |

---

## 七、Score SDE 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **理论统一** | 将 DDPM、Score-based、Langevin MCMC 统一到一个框架 |
| **灵活性强** | 支持多种 SDE 类型（VP、VE、sub-VP 等） |
| **采样多样** | 同时支持随机采样和确定性采样 |
| **可控精度** | 可通过 predictor-corrector 平衡速度和质量 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **仍需设计 $g(t)$** | 扩散系数仍是预设的超参数 |
| **公式更抽象** | SDE 比离散链更难理解 |
| **训练未简化** | 本质上还是 Score Matching / Noise Prediction |
| **实现复杂度高** | 需要数值积分 SDE/ODE |

---

## 八、三代演进脉络

```
DDPM (2020):                        DDIM (2021):                     Score SDE (2021):
┌────────────────┐                  ┌────────────────┐              ┌──────────────────────┐
│ 离散马尔可夫链  │                  │ 离散非马尔可夫  │              │ 连续 SDE/ODE         │
│                │                  │                │              │                      │
│ L = ‖ε - εθ‖²  │                  │ L = ‖ε - εθ‖²  │              │ L = λ·‖sθ - ∇log p‖²│
│                │                  │                │              │                      │
│ 学: εθ (噪声)   │                  │ 学: εθ (噪声)   │              │ 学: sθ (score)       │
│                │                  │                │              │                      │
│ 1000 步采样    │  去掉随机+加速  │ 50 步采样      │  统一+连续化  │ 可变步数采样          │
│                │                  │                │              │                      │
│ 设计 β_t       │                  │ 复用 β_t       │              │ 设计 g(t)             │
└────────────────┘                  └────────────────┘              └──────────────────────┘
  开创者                              加速者                            统一者
```

> **下一章：Flow Matching —— 如何彻底摆脱噪声调度的束缚，实现最简洁的训练范式？**
