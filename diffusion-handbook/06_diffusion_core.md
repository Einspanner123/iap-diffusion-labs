# 第6章 扩散模型的核心理论：前向过程与反向过程

> 前面五章我们学习了所有必要的数学工具。现在，让我们把它们组合起来，推导扩散模型的核心理论。这一章是整个手册的"心脏"——你将看到高斯分布、贝叶斯定理、KL 散度、变分推断如何协同工作，构建出扩散模型的完整数学框架。

---

## 6.1 全景概览

扩散模型由两个过程组成：

```
前向过程（加噪）：x₀ → x₁ → x₂ → ... → x_T
                  数据    逐步加噪声      纯噪声

反向过程（去噪）：x_T → x_{T-1} → ... → x₁ → x₀
                  纯噪声  逐步去噪          数据
```

**核心问题**：
1. 前向过程如何定义？（第 6.2 节）
2. 反向过程如何推导？（第 6.3 节）
3. 如何训练反向过程？（第 6.4 节）
4. 如何从噪声生成数据？（第 6.5 节）

---

## 6.2 前向过程：从数据到噪声

### 6.2.1 定义

前向过程是一个**马尔可夫链**，从真实数据 $x_0$ 出发，每一步加入少量高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

等价地，用重参数化表示：

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

**参数含义**：
- $\beta_t \in (0, 1)$：噪声调度，控制第 $t$ 步加入的噪声量
- $\sqrt{1-\beta_t}$：信号保留系数
- $\sqrt{\beta_t}$：噪声系数

### 6.2.2 关键推导：从 $x_0$ 直接到 $x_t$

**问题**：如果按定义，从 $x_0$ 得到 $x_t$ 需要迭代 $t$ 步。能不能一步到位？

**定义辅助变量**：

$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

**推导**（用数学归纳法）：

**第一步**：$t = 1$

$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_0$$

**假设**：$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \bar{\epsilon}_{t-2}$

**递推**：

$$\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \bar{\epsilon}_{t-2}) + \sqrt{1-\alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})} \bar{\epsilon}_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1}
\end{aligned}$$

**关键步骤**：合并两个独立高斯噪声

$$\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})} \bar{\epsilon}_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1} \sim \mathcal{N}(0, (\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t))I)$$

验证方差：

$$\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$$

因此：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**这就是扩散模型最重要的公式之一——重参数化技巧的直接结果！**

### 6.2.3 信噪比（SNR）

定义信噪比：

$$\text{SNR}(t) = \frac{\text{信号功率}}{\text{噪声功率}} = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

| 时间步 | $\bar{\alpha}_t$ | SNR | 含义 |
|--------|-----------------|-----|------|
| $t = 0$ | $\approx 1$ | $\gg 1$ | 几乎纯信号 |
| $t = T/2$ | 中间值 | $\approx 1$ | 信号与噪声相当 |
| $t = T$ | $\approx 0$ | $\ll 1$ | 几乎纯噪声 |

SNR 是理解扩散模型行为的关键指标。好的噪声调度应该让 SNR 平滑地从高到低变化。

---

## 6.3 反向过程：从噪声到数据

### 6.3.1 问题：真实的反向分布不可计算

前向过程 $q(x_t | x_{t-1})$ 是我们定义的，完全已知。但反向过程 $q(x_{t-1} | x_t)$ 需要知道整个数据分布——这是不可计算的。

**为什么不可计算？** 由贝叶斯定理：

$$q(x_{t-1} | x_t) = \frac{q(x_t | x_{t-1}) q(x_{t-1})}{q(x_t)}$$

$q(x_t)$ 需要对所有可能的数据积分，在高维空间中不可行。

### 6.3.2 解决方案：用神经网络近似反向分布

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

神经网络 $\mu_\theta$ 学习预测反向过程的均值。$\sigma_t^2$ 可以是固定的或可学习的。

### 6.3.3 关键推导：给定 $x_0$ 时的后验分布

虽然 $q(x_{t-1} | x_t)$ 不可计算，但 $q(x_{t-1} | x_t, x_0)$ 是可以计算的！

利用贝叶斯定理：

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

（利用了马尔可夫性：$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$）

三个分布都是高斯分布：

$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$$
$$q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1}) I)$$
$$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

**对数展开与配方**（这是第 4 章贝叶斯定理的完整应用）：

$$\log q(x_{t-1}|x_t, x_0) = \log q(x_t|x_{t-1}) + \log q(x_{t-1}|x_0) - \log q(x_t|x_0) + C$$

只保留与 $x_{t-1}$ 有关的项：

$$= -\frac{1}{2}\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)x_{t-1}^2 + \left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)x_{t-1} + C$$

配方后得到高斯分布的参数：

**方差**：

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

**均值**：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0$$

### 6.3.4 将均值改写为关于噪声的形式

这是 DDPM 最巧妙的一步。令 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，则：

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$$

代入 $\tilde{\mu}_t$ 并化简（过程见第 4 章练习 3 的推广）：

$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)$$

**这个形式的意义**：均值完全由 $x_t$ 和噪声 $\epsilon$ 决定。如果我们能预测噪声 $\epsilon$，就能计算均值，从而定义反向过程！

---

## 6.4 损失函数的推导

### 6.4.1 从 ELBO 出发

由第 5 章的变分推断框架：

$$-\text{ELBO} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0$$

其中 $L_T$ 是常数，$L_0$ 影响很小，核心是 $L_{t-1}$：

$$L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$

### 6.4.2 KL 散度的计算

两个高斯分布的 KL 散度（方差相同时）：

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t - \mu_\theta\|^2$$

将均值用噪声预测表示：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)$$

$$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

代入：

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t} \cdot \frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

### 6.4.3 简化损失函数

Ho et al. (2020) 发现，**忽略权重系数**，使用简化损失效果更好：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**为什么简化损失更好？**

VLB 损失中的权重 $w_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)}$ 在不同时间步差异极大：

| 时间步 | SNR | $w_t$ | VLB 行为 | 简化损失行为 |
|--------|-----|-------|---------|-------------|
| 低 $t$（弱噪声） | 高 | 极大 | 过度关注细节 | 等权训练 |
| 高 $t$（强噪声） | 低 | 极小 | 高噪声步几乎不训练 | 等权训练 |

简化损失让所有时间步获得等权训练，高噪声步的全局结构学习更充分，生成质量更优。

---

## 6.5 采样算法

### 6.5.1 训练算法

```
重复以下步骤：
1. 采样数据 x₀ ~ q(x₀)
2. 采样时间步 t ~ Uniform({1, ..., T})
3. 采样噪声 ε ~ N(0, I)
4. 计算噪声图像 x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
5. 计算损失 L = ‖ε - ε_θ(x_t, t)‖²
6. 更新参数 θ ← θ - η · ∇_θ L
```

### 6.5.2 采样算法（DDPM）

```
1. 采样 x_T ~ N(0, I)
2. 对 t = T, T-1, ..., 1：
   a. 采样 z ~ N(0, I)（如果 t > 1）
   b. 计算 x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + σ_t · z
3. 输出 x₀
```

其中 $\sigma_t^2 = \tilde{\beta}_t$（后验方差）或 $\sigma_t^2 = \beta_t$（简化方差）。

---

## 6.6 数学工具的完整回顾

让我们回顾一下，推导扩散模型核心理论用到了哪些数学工具：

| 推导步骤 | 使用的数学工具 |
|---------|--------------|
| 前向过程的定义 | 高斯分布、重参数化技巧 |
| 从 $x_0$ 直接到 $x_t$ | 独立高斯的和、数学归纳法 |
| 后验分布的推导 | 贝叶斯定理、高斯分布的对数展开、配方 |
| 均值改写为噪声形式 | 代数化简、重参数化 |
| ELBO 推导 | 变分推断、Jensen 不等式 |
| KL 散度计算 | 两个高斯的 KL 散度公式 |
| 简化损失 | 期望、范数 |
| 训练算法 | 蒙特卡洛采样、梯度下降 |

**看到了吗？** 前 5 章学习的每一个数学工具，都在这里找到了用武之地。这不是巧合——扩散模型的设计本身就是由这些数学工具塑造的。

---

## 练习与思考

1. **前向过程**：设 $\beta_1 = 0.01$，$\beta_2 = 0.02$，$\beta_3 = 0.03$。计算 $\alpha_t$ 和 $\bar{\alpha}_t$（$t = 1, 2, 3$）。

2. **后验分布**：设 $\bar{\alpha}_2 = 0.95$，$\bar{\alpha}_3 = 0.90$，$\beta_3 = 0.05$。计算 $\tilde{\beta}_3$ 和 $\tilde{\mu}_3$ 的表达式。

3. **损失函数**：设 $\alpha_t = 0.95$，$\bar{\alpha}_t = 0.80$，$\beta_t = 0.05$。计算 VLB 损失中的权重 $w_t$。

4. **思考题**：前向过程为什么选择"逐步加噪"而不是"一步加噪"？如果直接 $x_T = \epsilon$，会有什么问题？（提示：考虑反向过程需要什么信息）
