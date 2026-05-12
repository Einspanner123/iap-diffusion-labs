# 第7章 DDPM：去噪扩散概率模型

> DDPM 是扩散模型的开山之作，让扩散模型重新成为生成模型的主流。这一章我们将深入 DDPM 的工程细节：噪声调度如何设计、三种参数化方式如何选择、采样算法如何实现，以及 DDPM 与 Score Matching 的深层联系。

---

## 7.1 DDPM 的完整框架回顾

DDPM（Ho et al., 2020）的核心框架：

| 组件 | 定义 |
|------|------|
| 前向过程 | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ |
| 反向过程 | $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$ |
| 训练目标 | $\mathcal{L}_{\text{simple}} = \mathbb{E}[\Vert\epsilon - \epsilon_\theta(x_t, t)\Vert^2]$ |
| 采样 | 从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪 |

---

## 7.2 噪声调度的设计

### 7.2.1 线性调度（Linear Schedule）

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

Ho et al. (2020) 使用 $\beta_{\min} = 10^{-4}$，$\beta_{\max} = 0.02$，$T = 1000$。

**特点**：
- 简单直观
- 前期 $\beta_t$ 小，后期 $\beta_t$ 大
- 问题：前期低频结构被破坏太快

### 7.2.2 余弦调度（Cosine Schedule）

Nichol & Dhariwal (2021) 提出更好的调度：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是偏移量，防止 $\beta_t$ 在 $t=0$ 时太小。

**为什么余弦调度更好？**

线性调度下，$\bar{\alpha}_t$ 在前期快速下降，意味着图像的低频结构（整体轮廓）在早期就被破坏。余弦调度让 $\bar{\alpha}_t$ 前期缓慢下降，保留了更多结构信息。

```
线性调度的 ᾱ_t 变化：        余弦调度的 ᾱ_t 变化：
1.0 ┤\                       1.0 ┤\
    | \                           |  \
    |  \                          |   \
    |   \___                      |    \___
    |       \_____                |        \_____
0.0 ┤             \_______   0.0 ┤             \_______
    0              T              0              T
    前期下降太快                    前期缓慢下降
```

### 7.2.3 从数学角度理解调度

$\bar{\alpha}_t$ 决定了信噪比 $\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$。好的调度应该让 SNR **单调递减**且**变化平滑**。

从 SDE 的角度看，$\beta(t)$ 对应连续时间的噪声强度。线性调度对应 $\beta(t)$ 为常数，余弦调度对应 $\beta(t)$ 随时间增加——这与物理中"加速扩散"的直觉一致。

---

## 7.3 三种参数化方式

### 7.3.1 ε-prediction（预测噪声）

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

网络 $\epsilon_\theta$ 直接预测加入的噪声 $\epsilon$。

**优点**：最直观——"预测加了多少噪声，然后减掉"
**缺点**：在高噪声时（大 $t$），信号很弱，预测噪声的梯度可能不稳定

### 7.3.2 x₀-prediction（预测原始数据）

$$\hat{x}_0 = f_\theta(x_t, t)$$

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} f_\theta(x_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

网络 $f_\theta$ 直接预测原始数据 $x_0$。

**优点**：直接预测目标，直觉清晰
**缺点**：高噪声时，$x_0$ 的预测不稳定（因为信号太弱）

### 7.3.3 v-prediction（预测速度）

Salimans & Ho (2022) 提出：

$$v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$$

网络预测 $v$，然后恢复 $\epsilon$ 或 $x_0$：

$$\epsilon = \sqrt{\bar{\alpha}_t} v + \sqrt{1-\bar{\alpha}_t} x_t / \sqrt{\bar{\alpha}_t}$$

**优点**：数值稳定性最好——$v$ 在高噪声和低噪声时都有合理的范围
**适用**：级联模型（如 Imagen）、高分辨率生成

### 7.3.4 三种参数化的对比

| 参数化 | 网络输出 | 数值稳定性 | 适用场景 |
|--------|---------|-----------|---------|
| ε-prediction | $\epsilon_\theta$ | 一般 | 最常见，DDPM 默认 |
| x₀-prediction | $\hat{x}_0$ | 较差 | 需要直接估计 $x_0$ |
| v-prediction | $v_\theta$ | 最好 | 级联模型、高分辨率 |

**数学上的统一视角**：三种参数化只是对同一个均值 $\mu_\theta$ 的不同参数化方式，它们在理论上等价，差异在于数值稳定性和训练动态。

---

## 7.4 噪声预测网络：UNet 与时间步嵌入

### 7.4.1 为什么需要时间步嵌入？

噪声预测网络 $\epsilon_\theta(x_t, t)$ 的权重在所有时间步上共享。如果不告诉网络当前的噪声水平，网络无法区分不同 $t$ 的 $x_t$——同样的 $x_t$ 在 $t=100$ 和 $t=900$ 时，噪声水平完全不同，需要不同的去噪策略。

### 7.4.2 正弦位置编码

采用与 Transformer 相同的正弦编码：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

这种编码的优势：
- 不同维度对应不同的频率，可以捕捉多尺度的噪声水平信息
- 相对位置可以通过线性变换表示，有利于网络学习

### 7.4.3 UNet 架构

UNet 是 DDPM 的标准骨干网络，特点：
- **编码器**：逐步下采样，提取多尺度特征
- **解码器**：逐步上采样，恢复空间分辨率
- **跳跃连接**：将编码器的特征直接传递给解码器，保留细节信息
- **时间嵌入融合**：在每个残差块中，将时间嵌入加到卷积特征上

$$h' = h + \text{Linear}(\text{TimeEmbed}(t))$$

---

## 7.5 σ_t 的选择

反向过程中的噪声标准差 $\sigma_t$ 有两种常见选择：

### 7.5.1 后验方差

$$\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

这是理论最优的选择，匹配真实后验的方差。

### 7.5.2 简化方差

$$\sigma_t^2 = \beta_t$$

这是近似选择，略大于后验方差。

### 7.5.3 可学习方差

Nichol & Dhariwal (2021) 让网络同时预测 $\sigma_t^2$，在 $\tilde{\beta}_t$ 和 $\beta_t$ 之间插值：

$$\sigma_t^2 = \exp(v \log \tilde{\beta}_t + (1-v) \log \beta_t)$$

其中 $v$ 是网络输出的插值系数。这种方式略微提升了对数似然，但对生成质量影响不大。

---

## 7.6 DDPM 与 Score Matching 的等价关系

### 7.6.1 Score 函数

Score 函数定义为对数概率密度的梯度：

$$s(x, t) = \nabla_x \log p_t(x)$$

它指向概率密度增长最快的方向。

### 7.6.2 噪声预测与 Score 的关系

在 VP-SDE（DDPM 的连续版本）中：

$$s_\theta(x, t) = -\frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar{\alpha}_t}}$$

**证明思路**：

由 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，给定 $x_0$ 时：

$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此，预测噪声 $\epsilon$ 等价于预测 Score（只差一个缩放因子）。

### 7.6.3 这个等价关系的意义

1. **统一了两个独立发展的领域**：DDPM（扩散概率模型）和 Score-Based 模型实际上是同一件事
2. **提供了新的视角**：从 Score 的角度理解扩散模型，可以自然地推广到 SDE 框架
3. **解释了为什么简化损失有效**：预测噪声的 MSE 等价于 Score Matching

---

## 7.7 DDPM 的优缺点

### 7.7.1 优点

1. **训练稳定**：不像 GAN 需要精心平衡生成器和判别器
2. **生成质量高**：FID 分数与 GAN 相当甚至更好
3. **多样性好**：不会出现 GAN 的模式坍塌问题
4. **理论优雅**：有完整的概率论框架支撑

### 7.7.2 缺点

1. **采样速度慢**：需要 1000 步去噪，每步都需要一次前向传播
2. **计算成本高**：生成一张图片需要几秒到几十秒
3. **对数似然不如自回归模型**：虽然生成质量好，但密度估计不是最优的

---

## 7.8 本章小结

DDPM 是扩散模型的"第一代"，它建立了完整的数学框架：

| 组件 | DDPM 的选择 |
|------|------------|
| 前向过程 | 离散马尔可夫链，线性/余弦噪声调度 |
| 反向过程 | 高斯近似，ε-prediction |
| 损失函数 | 简化损失 $\Vert\epsilon - \epsilon_\theta\Vert^2$ |
| 骨干网络 | UNet + 时间步嵌入 |
| 采样步数 | 1000 步 |

DDPM 的主要问题是**采样太慢**。下一章我们将看到 DDIM 如何通过确定性采样将速度提升 20 倍。

---

## 练习与思考

1. **余弦调度**：验证当 $s = 0.008$，$T = 1000$ 时，余弦调度下 $\beta_1$ 和 $\beta_{1000}$ 的值。

2. **参数化转换**：给定 $\epsilon_\theta(x_t, t) = 0.5$，$\bar{\alpha}_t = 0.8$，$x_t = 1.0$。计算对应的 $\hat{x}_0$ 和 $v$。

3. **Score 关系**：验证 $s_\theta = -\epsilon_\theta / \sqrt{1-\bar{\alpha}_t}$ 在 $\bar{\alpha}_t = 0.5$ 时的缩放因子。

4. **思考题**：为什么 v-prediction 比 ε-prediction 更稳定？从 $v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$ 出发，分析当 $t$ 很大和很小时 $v$ 的行为。
