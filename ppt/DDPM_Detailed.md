# 第一代：DDPM (Denoising Diffusion Probabilistic Models)

> **论文：** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
>
> **地位：** 扩散模型的开山之作，让扩散模型重新成为生成模型的主流

---

## 一、核心思想

### 1.1 直觉：从清晰到模糊，再从模糊到清晰

```
正向过程（加噪）：                    反向过程（去噪）：
猫的图片  ──加噪声──→  纯噪声         纯噪声  ──去噪──→  猫的图片
t=0        t=1          t=2...T       T           ...2     t=0

类比：
把一张照片慢慢"雾化"，直到完全看不清。
然后训练一个神经网络，学会从"有雾的照片"中恢复原图。
```

### 1.2 为什么叫"概率模型"？

DDPM 的全称是 **Denoising Diffusion Probabilistic Models**：

| 单词 | 含义 |
|------|------|
| **Denoising** | 去噪——核心任务是预测并去除噪声 |
| **Diffusion** | 扩散——像墨水在水中扩散一样，噪声逐步扩散到图像中 |
| **Probabilistic** | 概率性——用概率分布描述每一步的状态 |
| **Models** | 模型 |

---

## 二、数学框架

### 2.1 前向过程（Forward Process / 加噪）

**定义：** 从真实数据 $x_0$ 出发，逐步添加高斯噪声，直到变成纯噪声 $x_T$。

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

#### 参数详解

| 符号 | 名称 | 形状 | 含义 |
|------|------|------|------|
| $x_0$ | 原始数据 | $(B, C, H, W)$ 或 $(B, d)$ | 真实的图片/数据样本 |
| $x_t$ | 第 $t$ 步的噪声数据 | 同上 | 加了 $t$ 步噪声后的数据 |
| $T$ | 总步数 | 标量 | 通常 = 1000 |
| $\beta_t$ | 噪声调度 | 标量序列 $(T,)$ | 第 $t$ 步添加的噪声方差 |
| $I$ | 单位矩阵 | $(d, d)$ | 保证各维度独立同分布 |
| $q(\cdot\|\cdot)$ | 前向条件分布 | — | 真实的（已知的）转移分布 |

#### $\beta_t$ 的含义

$\beta_t \in (0, 1)$ 控制第 $t$ 步加入多少噪声：

- $\beta_t$ 小 → 每步加少量噪声 → 需要很多步才能变纯噪声
- $\beta_t$ 大 → 每步加大量噪声 → 很快就变纯噪声

**常用调度：**

$$\beta_t = \text{linear}(10^{-4}, 0.02) \quad \text{或} \quad \beta_t = \cos^2\left(\frac{t/T \cdot \pi}{2}\right)$$

#### 重要推导：直接从 $x_0$ 到 $x_t$

由于高斯分布的可叠加性，我们可以**跳过中间步骤**，直接计算 $q(x_t|x_0)$：

定义两个辅助变量：

$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

| 变量 | 含义 |
|------|------|
| $\alpha_t = 1 - \beta_t$ | 第 $t$ 步保留的数据比例 |
| $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$ | 前 $t$ 步累积保留的数据比例 |

**证明：**

$$\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I) \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
&\vdots \\
&= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \underbrace{\epsilon}_{\sim \mathcal{N}(0, I)}
\end{aligned}$$

最后一步利用了：独立正态随机变量的线性组合仍然是正态分布。

> 🎯 **关键公式（重参数化技巧）：**
>
> $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
>
> 这意味着：**给定 $x_0$ 和 $t$，可以直接采样出 $x_t$，无需迭代！**

---

### 2.2 反向过程（Reverse Process / 去噪）

**定义：** 从纯噪声 $x_T$ 出发，逐步去除噪声，恢复原始数据 $x_0$。

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

#### 参数详解

| 符号 | 名称 | 含义 |
|------|------|------|
| $p_\theta(\cdot\|\cdot)$ | 反向条件分布 | 神经网络学习的近似分布 |
| $\mu_\theta(x_t, t)$ | 预测均值 | 神经网络输出第 $t$ 步的去噪均值 |
| $\sigma_t$ | 噪声标准差 | 通常设为固定值或 $\tilde{\beta}_t$ |
| $\theta$ | 神经网络参数 | 如 UNet 的权重 |

**注意：** 反向过程的精确分布 $q(x_{t-1}|x_t)$ 是**不可计算的**（需要知道整个数据分布），所以用神经网络 $p_\theta$ 来**近似**它。

---

### 2.3 训练目标：最大化对数似然

**理论最优损失（ELBO）：**

$$\mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)] \leq \sum_{t=1}^{T} D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_T)$$

这个 ELBO（Evidence Lower Bound）太复杂，难以直接优化。**DDPM 的核心贡献是将其简化为可计算的损失函数。**

#### 简化后的损失函数

经过一系列推导（见下方），DDPM 的最终损失简化为：

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\underbrace{x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon}_{\text{输入}}, t)\|^2\right]}$$

#### 完整推导链

**Step 1：写出反向分布的真实形式**

利用贝叶斯定理：

$$q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

由于前向过程是马尔可夫的：$q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1})$

**Step 2：代入高斯分布的具体形式**

$$\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) \\
q(x_{t-1}|x_0) &= \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})I) \\
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
\end{aligned}$$

**Step 3：利用高斯分布的乘积性质**

两个高斯分布的条件分布仍然是高斯分布。经过配方可得：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

所以真实的反向分布是：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

**Step 4：将均值改写为关于噪声的形式**

这是 DDPM 最巧妙的一步。令 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，则 $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$

代入 $\tilde{\mu}_t$：

$$\begin{aligned}
\tilde{\mu}_t &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t \\
&= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)
\end{aligned}$$

**Step 5：设计神经网络的预测目标**

让神经网络预测噪声 $\epsilon$ 而不是均值 $\tilde{\mu}_t$：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

**Step 6：KL 散度简化**

$$D_{KL}(q \| p_\theta) = \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t - \mu_\theta\|^2 + C$$

代入后得到：

$$\mathcal{L}_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

忽略常数系数，得到最终简化的损失：

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]}$$

---

### 2.4 采样算法

**算法：DDPM 采样（反向过程）**

| 步骤 | 操作 |
|------|------|
| **输入** | 神经网络 $\epsilon_\theta$，噪声调度 $\{\beta_t\}$，总步数 $T$ |
| **初始化** | $x_T \sim \mathcal{N}(0, I)$ |
| **循环** | 对 $t = T, T-1, \ldots, 1$： |
| ① | 计算 $\hat{\mu}_t = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$ |
| ② | 如果 $t > 1$：采样 $z \sim \mathcal{N}(0, I)$；否则 $z = 0$ |
| ③ | $x_{t-1} = \hat{\mu}_t + \sigma_t z$，其中 $\sigma_t = \tilde{\beta}_t$ 或 $\sigma_t = \beta_t$ |
| **输出** | $x_0$（生成的数据） |

> ⚠️ 注意第②步中的**随机项** $\sigma_t z$：这就是 SDE 的来源！每一步都加入了新的随机噪声。

---

### 2.5 完整变量表

| 变量 | 类型 | 定义域 | 说明 |
|------|------|--------|------|
| $x_0$ | 数据向量 | $\mathbb{R}^d$ | 原始数据（如图片展平后的像素值） |
| $x_t$ | 噪声向量 | $\mathbb{R}^d$ | 第 $t$ 步加噪后的数据 |
| $T$ | 正整数 | 通常 1000 | 总扩散步数 |
| $t$ | 正整数 | $[1, T]$ | 当前时间步 |
| $\beta_t$ | 标量 | $(0, 1)$ | 第 $t$ 步的噪声方差（超参数） |
| $\alpha_t$ | 标量 | $(0, 1)$ | $1 - \beta_t$，数据保留比例 |
| $\bar{\alpha}_t$ | 标量 | $(0, 1)$ | $\prod_{s=1}^{t} \alpha_s$，累积保留比例 |
| $\epsilon$ | 噪声向量 | $\mathbb{R}^d$ | 标准高斯噪声 $\mathcal{N}(0, I)$ |
| $\epsilon_\theta$ | 神经网络 | $\mathbb{R}^d \times [1,T] \to \mathbb{R}^d$ | 噪声预测器（通常是 UNet） |
| $\theta$ | 参数集 | — | 神经网络的全部可学习参数 |
| $\sigma_t$ | 标量 | $(0, 1)$ | 反向过程中的噪声标准差 |
| $\tilde{\beta}_t$ | 标量 | $(0, 1)$ | $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ |

---

## 三、DDPM 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| 训练稳定 | 损失函数简单（MSE），不会崩溃 |
| 生成质量高 | 在多个基准上超越 GAN |
| 无需对抗训练 | 不像 GAN 那样有模式崩塌问题 |
| 理论优雅 | 有明确的概率解释 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **采样慢** | 需要 1000 步迭代 |
| **需要设计 $\beta_t$** | 噪声调度是超参数，影响效果 |
| **随机性不可控** | 输入相同噪声，每次结果不同 |
| **公式复杂** | 大量 $\alpha$, $\beta$, $\bar{\alpha}$ 等 |

---

## 四、与后续方法的对比预览

| 特征 | DDPM | 后续方法改进方向 |
|------|------|------------------|
| 数学形式 | SDE（有随机项） | → ODE（去掉随机项） |
| 采样步数 | 1000 步 | → 50 步甚至更少 |
| 噪声调度 | 需要设计 $\beta_t$ | → 自动/无需设计 |
| 学习目标 | 预测噪声 $\epsilon$ | → 直接回归向量场 |
| 公式复杂度 | 高（大量辅助变量） | → 低（简洁统一） |

> **下一章：DDIM —— 如何去掉 DDPM 中的随机性，实现快速确定性采样？**
