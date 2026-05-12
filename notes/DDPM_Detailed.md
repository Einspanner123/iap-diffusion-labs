# 第一代：DDPM (Denoising Diffusion Probabilistic Models)

> **论文：** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
>
> **地位：** 扩散模型的开山之作，让扩散模型重新成为生成模型的主流

---

## 目录

1. [核心思想](#一核心思想)
2. [数学框架](#二数学框架)
3. [前向过程的完整推导](#三前向过程的完整推导)
4. [反向过程的完整推导](#四反向过程的完整推导)
5. [损失函数的完整推导](#五损失函数的完整推导)
6. [三种参数化方式](#六三种参数化方式)
7. [噪声调度的设计](#七噪声调度的设计)
8. [采样算法](#八采样算法)
9. [DDPM 与 Score Matching 的等价关系](#九ddpm-与-score-matching-的等价关系)
10. [完整变量表](#十完整变量表)
11. [优缺点总结](#十一优缺点总结)
12. [实践代码示例](#十二实践代码示例)
13. [与后续方法的对比预览](#十三与后续方法的对比预览)

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

### 1.3 DDPM 的两大核心假设

**假设 1：前向过程是高斯马尔可夫链**

每一步只依赖前一步（马尔可夫性），且转移分布是高斯分布：

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**假设 2：反向过程也是高斯的**

虽然真实的反向分布 $q(x_{t-1}|x_t)$ 非常复杂（需要知道整个数据分布），但 DDPM 假设它可以用高斯分布近似：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

> 💡 **为什么可以假设反向过程是高斯的？** 因为当 $\beta_t$ 足够小时，前向过程的一步变化很小，反向分布确实接近高斯。这是扩散模型理论的基石之一。

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
| $q(\cdot \mid \cdot)$ | 前向条件分布 | — | 真实的（已知的）转移分布 |

#### $\beta_t$ 的含义

$\beta_t \in (0, 1)$ 控制第 $t$ 步加入多少噪声：

- $\beta_t$ 小 → 每步加少量噪声 → 需要很多步才能变纯噪声
- $\beta_t$ 大 → 每步加大量噪声 → 很快就变纯噪声

**常用调度：**

$$\beta_t = \text{linear}(10^{-4}, 0.02) \quad \text{或} \quad \beta_t = \cos^2\left(\frac{t/T \cdot \pi}{2}\right)$$

---

### 2.2 反向过程（Reverse Process / 去噪）

**定义：** 从纯噪声 $x_T$ 出发，逐步去除噪声，恢复原始数据 $x_0$。

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

#### 参数详解

| 符号 | 名称 | 含义 |
|------|------|------|
| $p_\theta(\cdot \mid \cdot)$ | 反向条件分布 | 神经网络学习的近似分布 |
| $\mu_\theta(x_t, t)$ | 预测均值 | 神经网络输出第 $t$ 步的去噪均值 |
| $\sigma_t$ | 噪声标准差 | 反向过程的噪声标准差（见下文详细讨论） |
| $\theta$ | 神经网络参数 | 如 UNet 的权重 |

**注意：** 反向过程的精确分布 $q(x_{t-1}|x_t)$ 是**不可计算的**（需要知道整个数据分布），所以用神经网络 $p_\theta$ 来**近似**它。

---

## 三、前向过程的完整推导

### 3.1 定义辅助变量

$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

| 变量 | 含义 |
|------|------|
| $\alpha_t = 1 - \beta_t$ | 第 $t$ 步保留的数据比例 |
| $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$ | 前 $t$ 步累积保留的数据比例 |

### 3.2 从 $x_0$ 直接到 $x_t$ 的推导

**目标：** 证明 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$

**Step 1：** 写出前向过程的递推关系

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

**Step 2：** 递推展开两步

$$\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}}_{\text{两个独立高斯的混合}}
\end{aligned}$$

**Step 3：** 合并两个独立高斯噪声

**关键定理：** 如果 $\epsilon_1 \sim \mathcal{N}(0, \sigma_1^2 I)$ 和 $\epsilon_2 \sim \mathcal{N}(0, \sigma_2^2 I)$ 独立，则：

$$\sigma_1 \epsilon_1 + \sigma_2 \epsilon_2 \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)I)$$

因此：

$$\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1} \sim \mathcal{N}(0, (\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t))I)$$

**Step 4：** 验证方差

$$\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$$

所以展开两步后：

$$x_t = \sqrt{\alpha_t\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}} \bar{\epsilon}_{t-2}$$

其中 $\bar{\epsilon}_{t-2} \sim \mathcal{N}(0, I)$ 是一个等价的单一噪声。

**Step 5：** 归纳到 $t$ 步

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**验证：**

- 均值：$\mathbb{E}[x_t] = \sqrt{\bar{\alpha}_t} x_0$ ✓
- 方差：$\text{Var}(x_t) = (1-\bar{\alpha}_t)I$ ✓

> 🎯 **关键公式（重参数化技巧）：**
>
> $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
>
> 这意味着：**给定 $x_0$ 和 $t$，可以直接采样出 $x_t$，无需迭代！**

### 3.3 重参数化技巧的直觉

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε

当 t=0: ᾱ₀≈1, x_t ≈ x₀           （纯数据）
当 t=T: ᾱ_T≈0, x_t ≈ ε           （纯噪声）
中间:    x_t 是数据和噪声的混合
```

| 时间 | $\bar{\alpha}_t$ | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | 含义 |
|------|------------------|-------------------------|---------------------------|------|
| $t=0$ | $\approx 1$ | $\approx 1$ | $\approx 0$ | 几乎纯数据 |
| $t=T/2$ | 中间值 | 中间值 | 中间值 | 数据和噪声各半 |
| $t=T$ | $\approx 0$ | $\approx 0$ | $\approx 1$ | 几乎纯噪声 |

---

## 四、反向过程的完整推导

### 4.1 后验分布 $q(x_{t-1}|x_t, x_0)$ 的推导

**这是 DDPM 最重要的推导之一。** 我们需要知道：给定 $x_t$ 和 $x_0$，$x_{t-1}$ 的分布是什么？

**利用贝叶斯定理：**

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0) \cdot q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

由于前向过程是马尔可夫的：$q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1})$

**代入三个高斯分布：**

$$\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) \\
q(x_{t-1}|x_0) &= \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})I) \\
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
\end{aligned}$$

### 4.2 高斯贝叶斯的完整计算

**关键定理：** 如果 $q(x|y) = \mathcal{N}(x; A y, \Sigma_1)$ 且 $q(y) = \mathcal{N}(y; \mu, \Sigma_2)$，则 $q(y|x) \propto q(x|y) \cdot q(y)$ 也是高斯分布。

**对每个维度独立计算（省略 $I$）：**

$$\log q(x_{t-1}|x_t, x_0) = \log q(x_t|x_{t-1}) + \log q(x_{t-1}|x_0) - \log q(x_t|x_0) + C$$

**展开各项（只保留与 $x_{t-1}$ 有关的项）：**

$$\begin{aligned}
\log q(x_t|x_{t-1}) &= -\frac{1}{2\beta_t}(x_t - \sqrt{\alpha_t}x_{t-1})^2 + C_1 \\
\log q(x_{t-1}|x_0) &= -\frac{1}{2(1-\bar{\alpha}_{t-1})}(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2 + C_2 \\
\log q(x_t|x_0) &= \text{与 } x_{t-1} \text{ 无关}
\end{aligned}$$

**合并 $x_{t-1}$ 的二次项和一次项：**

$$\log q(x_{t-1}|x_t, x_0) = -\frac{1}{2}\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)x_{t-1}^2 + \left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)x_{t-1} + C$$

**配方得到高斯分布的参数：**

**精度（方差的倒数）：**

$$\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

利用 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$：

$$\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + \beta_t = \alpha_t - \bar{\alpha}_t + 1 - \alpha_t = 1 - \bar{\alpha}_t$$

因此：

$$\boxed{\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$$

**均值：**

$$\tilde{\mu}_t = \tilde{\beta}_t \left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)$$

代入 $\tilde{\beta}_t$：

$$\tilde{\mu}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \cdot \frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0$$

$$\boxed{\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0}$$

**最终结果：**

$$\boxed{q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)}$$

### 4.3 将均值改写为关于噪声的形式

**这是 DDPM 最巧妙的一步。** 令 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，则：

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$$

代入 $\tilde{\mu}_t$：

$$\begin{aligned}
\tilde{\mu}_t &= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}} \\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} x_t - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t\sqrt{1-\bar{\alpha}_t}}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \epsilon
\end{aligned}$$

**合并 $x_t$ 的系数：**

利用 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$，即 $\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}$：

$$\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} + \frac{\beta_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}$$

$$= \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} = \frac{1-\bar{\alpha}_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}}$$

**$\epsilon$ 的系数：**

$$-\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t\sqrt{1-\bar{\alpha}_t}}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} = -\frac{\beta_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} \cdot \sqrt{1-\bar{\alpha}_t} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}$$

**最终结果：**

$$\boxed{\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)}$$

---

## 五、损失函数的完整推导

### 5.1 ELBO 推导

**目标：** 最大化数据的对数似然 $\log p_\theta(x_0)$

$$\begin{aligned}
\log p_\theta(x_0) &= \log \int p_\theta(x_{0:T}) \mathrm{d}x_{1:T} \\
&= \log \int \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} q(x_{1:T}|x_0) \mathrm{d}x_{1:T} \\
&\geq \int q(x_{1:T}|x_0) \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \mathrm{d}x_{1:T} \quad \text{(Jensen 不等式)} \\
&= \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -\mathcal{L}_{\text{VLB}}
\end{aligned}$$

**展开 $\mathcal{L}_{\text{VLB}}$：**

$$\mathcal{L}_{\text{VLB}} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}$$

- $L_T$：常数（前向过程的终态与先验的 KL，不依赖 $\theta$）
- $L_0$：重建项
- $L_{t-1}$：每一步的去噪 KL 散度

### 5.2 KL 散度的计算

对于两个高斯分布 $q = \mathcal{N}(\mu_1, \sigma_1^2 I)$ 和 $p = \mathcal{N}(\mu_2, \sigma_2^2 I)$：

$$D_{KL}(q \| p) = \frac{d}{2}\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{d}{2}\left(\frac{\sigma_1^2}{\sigma_2^2} - 1\right) + \frac{\|\mu_1 - \mu_2\|^2}{2\sigma_2^2}$$

当 $\sigma_2^2 = \sigma_1^2 = \tilde{\beta}_t$（DDPM 的选择）时，KL 散度简化为：

$$D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) = \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t - \mu_\theta\|^2$$

### 5.3 设计神经网络的预测目标

让神经网络预测噪声 $\epsilon$ 而不是均值 $\tilde{\mu}_t$：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

代入 KL 散度：

$$\begin{aligned}
L_{t-1} &= \frac{1}{2\tilde{\beta}_t}\left\|\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right) - \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right)\right\|^2 \\
&= \frac{1}{2\tilde{\beta}_t} \cdot \frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2
\end{aligned}$$

定义权重 $w_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)}$，则：

$$L_{t-1} = w_t \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

### 5.3b L₀ 重建项的工程处理

$L_0 = -\log p_\theta(x_0|x_1)$ 是 VLB 中的重建项。对于归一化到 $[-1, 1]$ 的图像，像素值本质上是离散的（整数像素值），但 $p_\theta(x_0|x_1)$ 被建模为连续高斯分布，不能直接计算离散值的概率。

**原论文的处理方式：** 将 $p_\theta(x_0|x_1)$ 建模为高斯分布的区间积分——即对每个像素值 $x_0^{(i)}$，计算其在 $[x_0^{(i)} - \frac{1}{255}, x_0^{(i)} + \frac{1}{255}]$ 区间内的概率质量。

**实践中的简化：** 使用简化损失 $\mathcal{L}_{\text{simple}}$ 时，通常直接忽略 $L_0$，因为：
- $L_0$ 对生成质量影响极小
- 仅在需要计算精确对数似然（如评估 NLL/bpd 指标）时才需处理
- 简化损失已经隐式地通过 $t=1$ 的噪声预测覆盖了重建信息

### 5.4 简化损失函数

Ho et al. (2020) 发现**忽略权重 $w_t$**（即设 $w_t = 1$）反而效果更好：

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]}$$

**为什么忽略权重反而更好？**

| 加权损失 $\mathcal{L}_{\text{VLB}}$ | 简化损失 $\mathcal{L}_{\text{simple}}$ |
|--------------------------------------|----------------------------------------|
| 理论上更优（直接优化 ELBO） | 实践中更好 |
| 权重 $w_t$ 在不同 $t$ 差异很大 | 所有时间步等权 |
| 高噪声步（大 $t$）权重小 | 高噪声步也获得足够训练 |
| 低噪声步（小 $t$）权重大 | 生成质量更高 |

> 💡 **直觉：** 简化损失让网络在所有噪声水平上都学好，而加权损失让网络过度关注低噪声步，忽略了高噪声步的全局结构。

**深入理解：权重 $w_t$ 与信噪比（SNR）的关系**

VLB 损失中的权重可以化简为：

$$w_t \approx \frac{\beta_t}{2(1-\bar{\alpha}_t)}$$

而信噪比定义为 $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$，因此 $w_t$ 与 SNR 成反比：

| 时间步 | SNR | $w_t$ | VLB 的行为 | 简化损失的行为 |
|--------|-----|-------|-----------|--------------|
| 低 $t$（弱噪声） | 高 | **极大** | 过度关注细节，忽略全局结构 | 等权训练 |
| 高 $t$（强噪声） | 低 | **极小** | 高噪声步几乎得不到训练 | 等权训练 |

**结论：** 简化损失给所有时间步等权，让高 $t$ 步的全局结构学习更充分，生成质量更优。虽然 $\mathcal{L}_{\text{simple}}$ 不直接优化 ELBO，但优化它的同时也会降低 VLB（两者梯度方向大致一致）。

---

## 六、三种参数化方式

### 6.1 ε-prediction（预测噪声）

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

**这是 DDPM 原论文的选择，也是最常见的。**

### 6.2 x₀-prediction（预测原始数据）

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} f_\theta(x_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

其中 $f_\theta(x_t, t)$ 直接预测 $x_0$。

**与 ε-prediction 的关系：**

$$\hat{x}_0 = f_\theta(x_t, t) = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

### 6.3 v-prediction（预测速度）

Salimans & Ho (2022) 提出：

$$v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$$

$$\mu_\theta(x_t, t) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{\sqrt{1-\bar{\alpha}_t}(1-\bar{\alpha}_t)} v_\theta(x_t, t)$$

**v-prediction 的优势：** 在高噪声和低噪声时数值更稳定，特别适合级联模型。

### 6.4 三种参数化的对比

| 参数化 | 网络输出 | 数值稳定性 | 适用场景 |
|--------|----------|------------|----------|
| ε-prediction | $\epsilon_\theta$ | 一般 | 最常见，DDPM 默认 |
| x₀-prediction | $\hat{x}_0$ | 较差（高噪声时不稳定） | 需要直接估计 $x_0$ |
| v-prediction | $v_\theta$ | **最好** | 级联模型、高分辨率 |

### 6.5 噪声预测网络的核心：时间步嵌入

**为什么需要时间步嵌入？** 噪声预测网络 $\epsilon_\theta(x_t, t)$ 的权重在所有时间步 $t$ 上共享。如果不显式告知网络当前的噪声水平，网络无法区分不同 $t$ 的 $x_t$——同样的 $x_t$ 在 $t=100$ 和 $t=900$ 时，噪声水平完全不同，需要不同的去噪策略。

**标准实现：** 采用正弦位置编码（与 Transformer 位置编码一致）对 $t$ 编码，再通过全连接层映射后，融合到 UNet 的每一个残差块中。

```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

**时间步嵌入的融合方式：** 在 UNet 的每个残差块中，将时间嵌入通过一个线性层映射后，加到卷积特征上：

$$h' = h + \text{Linear}(\text{TimeEmbed}(t))$$

这样每个残差块都能感知当前的噪声水平，从而自适应地调整去噪策略。

---

## 七、噪声调度的设计

### 7.1 线性调度（Linear Schedule）

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

Ho et al. (2020) 使用 $\beta_{\min} = 10^{-4}$，$\beta_{\max} = 0.02$。

**特点：** 简单，但早期破坏低频结构太快。

### 7.2 余弦调度（Cosine Schedule）

Nichol & Dhariwal (2021) 提出：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是偏移量，防止 $\beta_t$ 在 $t=0$ 时太小。

**特点：** 更平滑的噪声增长，效果通常优于线性调度。

### 7.3 调度对 $\bar{\alpha}_t$ 的影响

| 调度类型 | $\bar{\alpha}_t$ 的变化 | 效果 |
|----------|------------------------|------|
| 线性 | 前期快速下降 | 早期就丢失低频信息 |
| 余弦 | 前期缓慢下降 | 保留更多结构信息 |

> 💡 **关键洞察：** $\bar{\alpha}_t$ 决定了信噪比（SNR）$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$。好的调度应该让 SNR 平滑地从高到低变化。

---

## 八、采样算法

### 8.1 σ_t 的选择

反向过程中的 $\sigma_t$ 有两种常见选择：

| 选择 | 公式 | 含义 |
|------|------|------|
| **后验方差** | $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ | 理论最优（匹配真实后验） |
| **简化方差** | $\sigma_t^2 = \beta_t$ | 简化近似 |

**Ho et al. (2020) 的发现：** 两种选择在生成质量上几乎没有区别！这是因为当 $T$ 足够大时，$\tilde{\beta}_t \approx \beta_t$。

**验证：** 当 $\beta_t$ 很小时，$\bar{\alpha}_{t-1} \approx \bar{\alpha}_t / \alpha_t \approx \bar{\alpha}_t$，因此：

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \approx \frac{1-\bar{\alpha}_t}{1-\bar{\alpha}_t}\beta_t = \beta_t$$

**但这个近似在以下情况下不成立：**

| 场景 | $\tilde{\beta}_t$ vs $\beta_t$ | 应选择 |
|------|-------------------------------|--------|
| 使用简化损失 $\mathcal{L}_{\text{simple}}$ | 差异对质量影响极小（只优化了均值） | 两者均可 |
| 需要精准优化对数似然 | 必须使用 $\tilde{\beta}_t$ 或可学习方差 | $\tilde{\beta}_t$ 或 $\Sigma_\theta(x_t, t)$ |
| 采样步数减少（如 1000→100） | 两者数值差距不可忽略 | **必须用 $\tilde{\beta}_t$** |

> ⚠️ **关键细节：** 简化损失 $\mathcal{L}_{\text{simple}}$ 只优化了均值 $\mu_\theta$，不优化方差 $\sigma_t^2$。因此使用简化损失时，$\sigma_t$ 的选择对生成质量影响极小。但若要精准优化对数似然，需将 $\sigma_t$ 设为可学习参数 $\Sigma_\theta(x_t, t)$，并使用 VLB 损失训练（Improved DDPM, Nichol & Dhariwal 2021）。

### 8.2 完整采样算法

**算法：DDPM 采样（反向过程）**

| 步骤 | 操作 |
|------|------|
| **输入** | 神经网络 $\epsilon_\theta$，噪声调度 $\{\beta_t\}$，总步数 $T$ |
| **初始化** | $x_T \sim \mathcal{N}(0, I)$ |
| **循环** | 对 $t = T, T-1, \ldots, 1$： |
| ① | 计算 $\hat{\mu}_t = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$ |
| ② | 如果 $t > 1$：采样 $z \sim \mathcal{N}(0, I)$；否则 $z = 0$ |
| ③ | $x_{t-1} = \hat{\mu}_t + \sigma_t z$ |
| **输出** | $x_0$（生成的数据） |

> ⚠️ 注意第②步中的**随机项** $\sigma_t z$：这就是 SDE 的来源！每一步都加入了新的随机噪声。DDIM 的核心改进就是去掉这个随机项。

---

## 九、DDPM 与 Score Matching 的等价关系

### 9.1 从噪声预测到 Score

**关键等式：** 预测噪声 $\epsilon_\theta(x_t, t)$ 等价于预测 score $\nabla_{x_t}\log p(x_t)$（差一个缩放因子）。

由 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，条件 score 为：

$$\nabla_{x_t}\log p(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此：

$$\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)$$

其中 $s_\theta(x_t, t) \approx \nabla_{x_t}\log p(x_t)$ 是 score 函数。

### 9.2 等价的意义

这个等价关系说明：
- **DDPM 本质上在学习 score**，只是用不同的参数化
- **Score SDE 可以看作 DDPM 的连续化推广**
- 两种方法在理论上等价，只是实现方式不同

### 9.3 DDPM 反向采样与朗之万动力学

将 $\epsilon_\theta = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)$ 代入反向过程的均值公式：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right) = \frac{1}{\sqrt{\alpha_t}}\left(x_t + \beta_t \cdot s_\theta(x_t, t)\right)$$

当 $\beta_t$ 很小时，$\frac{1}{\sqrt{\alpha_t}} = \frac{1}{\sqrt{1-\beta_t}} \approx 1 + \frac{\beta_t}{2}$，因此：

$$\tilde{\mu}_t \approx x_t + \beta_t \cdot s_\theta(x_t, t) + O(\beta_t^2)$$

而反向采样为 $x_{t-1} = \tilde{\mu}_t + \sigma_t z$，即：

$$x_{t-1} \approx x_t + \beta_t \cdot \nabla_{x_t}\log p(x_t) + \sqrt{\beta_t} \cdot z$$

**这正是朗之万动力学（Langevin Dynamics）** 的离散形式：

$$x' = x + \frac{\epsilon}{2}\nabla_x \log p(x) + \sqrt{\epsilon} \cdot z$$

其中步长 $\epsilon$ 对应 $\beta_t$。朗之万动力学是带噪声的梯度上升——沿着数据分布的 score（对数概率梯度）更新，同时加入高斯噪声保证采样遍历性。这也从另一个角度印证了 DDPM 与 Score Matching 的等价性。

---

## 十、完整变量表

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
| $\tilde{\beta}_t$ | 标量 | $(0, 1)$ | $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$，后验方差 |
| $\tilde{\mu}_t$ | 向量 | $\mathbb{R}^d$ | 后验均值（给定 $x_t, x_0$） |
| $w_t$ | 标量 | $(0, \infty)$ | VLB 损失的权重系数 |

---

## 十一、优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| 训练稳定 | 损失函数简单（MSE），不会崩溃 |
| 生成质量高 | 在多个基准上超越 GAN |
| 无需对抗训练 | 不像 GAN 那样有模式崩塌问题 |
| 理论优雅 | 有明确的概率解释和 ELBO 推导 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **采样慢** | 需要 1000 步迭代 |
| **需要设计 $\beta_t$** | 噪声调度是超参数，影响效果 |
| **随机性不可控** | 输入相同噪声，每次结果不同 |
| **公式复杂** | 大量 $\alpha$, $\beta$, $\bar{\alpha}$ 等辅助变量 |
| **简化损失非最优** | $\mathcal{L}_{\text{simple}}$ 不直接优化 ELBO |

---

## 十二、实践代码示例

### 12.1 DDPM 训练

```python
import torch
import torch.nn as nn

class DDPMScheduler:
    def __init__(self, T=1000, beta_schedule='linear'):
        self.T = T

        if beta_schedule == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
            self.alpha = 1.0 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float32)
            f_t = torch.cos((steps / T + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = f_t / f_t[0]
            self.alpha = alpha_bar[1:] / alpha_bar[:-1]
            self.beta = torch.clip(1 - self.alpha, 1e-4, 0.9999)
            self.alpha_bar = alpha_bar[1:]

class DDPMTrainer:
    def __init__(self, model, scheduler, lr=1e-4):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    def train_step(self, x_0):
        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(1, self.scheduler.T + 1, (B,), device=device)

        eps = torch.randn_like(x_0)

        alpha_bar = self.scheduler.alpha_bar.to(device)
        ab = alpha_bar[t - 1].view(-1, *([1] * (x_0.ndim - 1)))
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * eps

        eps_pred = self.model(x_t, t)

        loss = nn.functional.mse_loss(eps_pred, eps)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
```

### 12.2 DDPM 采样

```python
@torch.no_grad()
def sample_ddpm(model, scheduler, shape, device='cuda', return_all_steps=False):
    alpha = scheduler.alpha.to(device)
    alpha_bar = scheduler.alpha_bar.to(device)
    beta = scheduler.beta.to(device)

    x = torch.randn(shape, device=device)
    all_steps = [x.cpu()] if return_all_steps else None

    for t in reversed(range(1, scheduler.T + 1)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        t_idx = t - 1

        eps_pred = model(x, t_batch)

        ab = alpha_bar[t_idx]
        b = beta[t_idx]
        a = alpha[t_idx]

        mu = (1 / torch.sqrt(a)) * (x - (b / torch.sqrt(1 - ab)) * eps_pred)

        if t > 1:
            sigma = torch.sqrt(beta[t_idx])
            z = torch.randn_like(x)
            x = mu + sigma * z
        else:
            x = mu

        if return_all_steps:
            all_steps.append(x.cpu())

    return (x, all_steps) if return_all_steps else x
```

---

## 十三、与后续方法的对比预览

| 特征 | DDPM | 后续方法改进方向 |
|------|------|------------------|
| 数学形式 | SDE（有随机项） | → ODE（去掉随机项） |
| 采样步数 | 1000 步 | → 50 步甚至更少 |
| 噪声调度 | 需要设计 $\beta_t$ | → 自动/无需设计 |
| 学习目标 | 预测噪声 $\epsilon$ | → 直接回归向量场 |
| 公式复杂度 | 高（大量辅助变量） | → 低（简洁统一） |
| 参数化方式 | ε-prediction | → v-prediction, 向量场 |

> **下一章：DDIM —— 如何去掉 DDPM 中的随机性，实现快速确定性采样？**
