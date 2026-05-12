# 第5章 信息论与变分推断基础

> 如何衡量两个概率分布之间的"距离"？如何用一个简单的分布去近似一个复杂的分布？这些问题的答案——KL 散度和变分推断——正是推导扩散模型训练目标的数学工具。

---

## 5.1 信息熵

### 5.1.1 信息的度量

信息论的核心问题：**一个事件包含多少"信息"？**

直觉上：
- "明天太阳从东方升起"——几乎不包含信息（确定性事件）
- "明天彩票中奖号码是 1234567"——包含大量信息（极小概率事件）

**信息量的定义**：事件 $x$ 的自信息

$$I(x) = -\log p(x)$$

- 概率越小，信息量越大
- 取对数是为了保证独立事件的信息量可加：$I(x, y) = I(x) + I(y)$

### 5.1.2 信息熵

信息熵是自信息的期望，衡量随机变量的"平均不确定性"：

$$H(X) = -\sum_x p(x) \log p(x) \quad \text{（离散）}$$

$$H(X) = -\int p(x) \log p(x) dx \quad \text{（连续，也叫微分熵）}$$

**直觉**：
- 确定性分布（$p(x) = 1$）：熵 = 0，没有不确定性
- 均匀分布：熵最大，不确定性最大
- 高斯分布：在给定方差的分布中，高斯分布的熵最大

**高斯分布的微分熵**：

$$H(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)$$

注意：熵只依赖方差 $\sigma^2$，不依赖均值 $\mu$。**方差越大，不确定性越大，熵越大。**

---

## 5.2 KL 散度

### 5.2.1 定义

KL 散度（Kullback-Leibler divergence）衡量两个分布 $p$ 和 $q$ 之间的"距离"：

$$D_{KL}(q \| p) = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)} dx$$

### 5.2.2 KL 散度的性质

1. **非负性**：$D_{KL}(q \| p) \geq 0$，等号当且仅当 $q = p$（几乎处处）
2. **不对称性**：$D_{KL}(q \| p) \neq D_{KL}(p \| q)$——所以 KL 散度不是真正的"距离"
3. **等于 0 当且仅当** $q = p$

**为什么不对称？** $D_{KL}(q \| p)$ 衡量的是"用 $p$ 来近似 $q$ 时损失的信息"。方向不同，损失不同。

### 5.2.3 两个高斯分布之间的 KL 散度

这是扩散模型中最常用的公式。对于 $q = \mathcal{N}(\mu_1, \sigma_1^2 I)$ 和 $p = \mathcal{N}(\mu_2, \sigma_2^2 I)$：

$$D_{KL}(q \| p) = \frac{d}{2}\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{d}{2}\left(\frac{\sigma_1^2}{\sigma_2^2} - 1\right) + \frac{\|\mu_1 - \mu_2\|^2}{2\sigma_2^2}$$

当 $\sigma_1^2 = \sigma_2^2$ 时（DDPM 的情况），KL 散度简化为：

$$D_{KL}(q \| p) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$$

**这就是 DDPM 损失函数的来源！** 训练目标就是最小化真实后验均值 $\tilde{\mu}_t$ 和网络预测均值 $\mu_\theta$ 之间的 KL 散度，而它恰好等于两者之差的平方。

### 5.2.4 KL 散度的推导（一维情况）

$$D_{KL}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx$$

代入两个高斯分布的密度函数：

$$q(x) = \frac{1}{\sqrt{2\pi}\sigma_1}\exp\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)$$

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma_2}\exp\left(-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right)$$

$$\log \frac{q(x)}{p(x)} = \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

取期望（在 $q$ 下）：

$$D_{KL} = \log\frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}$$

$$= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} - \frac{1}{2} + \frac{(\mu_1 - \mu_2)^2}{2\sigma_2^2}$$

推广到 $d$ 维（各维度独立）就得到前面的公式。

---

## 5.3 变分推断与 ELBO

### 5.3.1 问题：后验分布不可计算

在很多问题中，我们需要计算后验分布 $p(z|x)$，但直接计算需要积分：

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|z)p(z) dz}$$

分母 $\int p(x|z)p(z) dz$ 在高维空间中无法解析计算——这就是**推断难题**。

### 5.3.2 解决思路：用简单分布近似后验

既然无法精确计算后验，那就**找一个简单的分布 $q_\phi(z)$ 来近似它**。

"最好的近似"就是让 $q_\phi$ 尽可能接近 $p(z|x)$，即最小化 $D_{KL}(q_\phi(z) \| p(z|x))$。

但这个 KL 散度本身也无法直接计算（因为需要知道 $p(z|x)$）！怎么办？

### 5.3.3 ELBO 的推导

**关键技巧**：从对数似然出发，利用 Jensen 不等式。

$$\log p(x) = \log \int p(x, z) dz = \log \int \frac{p(x, z)}{q_\phi(z)} q_\phi(z) dz$$

利用 Jensen 不等式（$\log$ 是凹函数）：

$$\log p(x) \geq \int q_\phi(z) \log \frac{p(x, z)}{q_\phi(z)} dz = \mathbb{E}_{q_\phi}\left[\log \frac{p(x, z)}{q_\phi(z)}\right]$$

右边就是 **ELBO（Evidence Lower Bound，证据下界）**：

$$\text{ELBO} = \mathbb{E}_{q_\phi}\left[\log \frac{p(x, z)}{q_\phi(z)}\right]$$

### 5.3.4 ELBO 与 KL 散度的关系

**最重要的等式**：

$$\log p(x) = \text{ELBO} + D_{KL}(q_\phi(z) \| p(z|x))$$

**证明**：

$$\begin{aligned}
\text{ELBO} &= \mathbb{E}_{q_\phi}\left[\log \frac{p(x, z)}{q_\phi(z)}\right] \\
&= \mathbb{E}_{q_\phi}\left[\log \frac{p(z|x) \cdot p(x)}{q_\phi(z)}\right] \\
&= \mathbb{E}_{q_\phi}[\log p(z|x)] + \log p(x) - \mathbb{E}_{q_\phi}[\log q_\phi(z)] \\
&= -D_{KL}(q_\phi \| p(z|x)) + \log p(x)
\end{aligned}$$

因此：

$$\log p(x) = \text{ELBO} + D_{KL}(q_\phi \| p(z|x))$$

**这个等式告诉我们**：
- $\log p(x)$ 是固定的（不依赖 $\phi$）
- 最大化 ELBO 等价于最小化 $D_{KL}(q_\phi \| p(z|x))$
- 当 $q_\phi = p(z|x)$ 时，KL = 0，ELBO = $\log p(x)$

### 5.3.5 ELBO 的分解

ELBO 可以进一步分解为：

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z)}[\log p(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z) \| p(z))}_{\text{正则项}}$$

- **重建项**：给定潜变量 $z$，重建数据 $x$ 的好坏
- **正则项**：近似后验 $q_\phi$ 与先验 $p(z)$ 的距离，防止 $q_\phi$ 偏离先验太远

---

## 5.4 变分推断在扩散模型中的应用

### 5.4.1 DDPM 的变分推断框架

在 DDPM 中：
- "数据" $x$ 对应 $x_0$（真实图像）
- "潜变量" $z$ 对应 $x_{1:T}$（所有中间噪声状态）
- "后验" $p(z|x)$ 对应 $q(x_{1:T}|x_0)$（前向过程）
- "近似后验" $q_\phi(z|x)$ 对应 $p_\theta(x_{1:T}|x_0)$（反向过程）

### 5.4.2 DDPM 的 ELBO 推导

$$\log p_\theta(x_0) \geq \text{ELBO} = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

展开分子和分母：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

$$q(x_{1:T}|x_0) = q(x_T|x_0) \prod_{t=1}^{T} q(x_t|x_{t-1}, x_0) = q(x_T|x_0) \prod_{t=1}^{T} q(x_t|x_{t-1})$$

代入 ELBO 并化简，得到：

$$-\text{ELBO} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T：先验匹配项} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}：去噪匹配项}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0：重建项}$$

**三项的含义**：

| 项 | 含义 | 是否依赖 $\theta$ |
|------|------|------------------|
| $L_T$ | 前向终态与先验的 KL | 否（常数） |
| $L_{t-1}$ | 真实后验与模型后验的 KL | **是** |
| $L_0$ | 重建质量 | 是 |

**训练目标**：最小化 $L_{t-1}$ 项——让模型学到的反向分布 $p_\theta(x_{t-1}|x_t)$ 尽可能接近真实后验 $q(x_{t-1}|x_t, x_0)$。

### 5.4.3 从 KL 散度到简化损失

由于 $q(x_{t-1}|x_t, x_0)$ 和 $p_\theta(x_{t-1}|x_t)$ 都是高斯分布，且方差相同，KL 散度简化为：

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t - \mu_\theta\|^2$$

将均值用噪声预测表示后：

$$L_{t-1} = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

忽略权重系数，得到简化损失：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**从信息论到深度学习**：整个推导链条是——

$$\text{对数似然} \to \text{ELBO} \to \text{KL 散度} \to \text{均方误差}$$

---

## 5.5 重参数化技巧

### 5.5.1 问题：梯度无法通过随机采样

在训练中，我们需要计算：

$$\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]$$

如果 $z \sim q_\phi(z) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$，直接采样的梯度是未定义的——因为采样操作不可微。

### 5.5.2 解决方案：重参数化

将随机性从参数中"分离"出来：

$$z = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

现在随机性来自 $\epsilon$（与 $\phi$ 无关），而 $\mu_\phi$ 和 $\sigma_\phi$ 是确定性的函数。梯度可以通过 $\mu_\phi$ 和 $\sigma_\phi$ 传递：

$$\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \epsilon)]$$

### 5.5.3 在扩散模型中的应用

DDPM 的前向过程天然使用了重参数化：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

训练时：
1. 采样 $\epsilon \sim \mathcal{N}(0, I)$
2. 计算 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
3. 计算 $\epsilon_\theta(x_t, t)$
4. 计算损失 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$
5. 梯度通过 $x_t$ 反向传播到 $\epsilon_\theta$ 的参数

**如果没有重参数化技巧**，第 2 步的采样操作会阻断梯度传播，整个训练就无法进行。

---

## 5.6 Jensen 不等式

### 5.6.1 定义

对于凸函数 $f$：

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

对于凹函数 $f$（如 $\log$）：

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

### 5.6.2 在 ELBO 推导中的应用

$$\log p(x) = \log \int p(x, z) dz = \log \int \frac{p(x, z)}{q(z)} q(z) dz = \log \mathbb{E}_{q}\left[\frac{p(x, z)}{q(z)}\right]$$

由于 $\log$ 是凹函数，由 Jensen 不等式：

$$\log \mathbb{E}_{q}\left[\frac{p(x, z)}{q(z)}\right] \geq \mathbb{E}_{q}\left[\log \frac{p(x, z)}{q(z)}\right] = \text{ELBO}$$

**Jensen 不等式是 ELBO 推导的唯一数学工具**——它把"对数放到积分里面"，产生了可以计算的 ELBO。

---

## 5.7 本章小结

| 信息论工具 | 在扩散模型中的角色 |
|-----------|-------------------|
| 信息熵 | 衡量分布的不确定性 |
| KL 散度 | 衡量真实后验与模型后验的距离 |
| 两个高斯的 KL 散度 | 推导 DDPM 的损失函数 |
| 变分推断与 ELBO | 推导训练目标的理论框架 |
| 重参数化技巧 | 让梯度能通过随机采样传播 |
| Jensen 不等式 | ELBO 推导的数学基础 |

---

## 练习与思考

1. **KL 散度计算**：设 $q = \mathcal{N}(0, 1)$，$p = \mathcal{N}(0, 4)$。计算 $D_{KL}(q \| p)$ 和 $D_{KL}(p \| q)$，验证两者不相等。

2. **ELBO 推导**：证明 $\log p(x) = \text{ELBO} + D_{KL}(q_\phi(z) \| p(z|x))$。（提示：展开 ELBO 的定义）

3. **重参数化**：设 $z = 3 + 2\epsilon$，$\epsilon \sim \mathcal{N}(0, 1)$。求 $z$ 的分布，并验证 $\frac{\partial z}{\partial 3} = 1$，$\frac{\partial z}{\partial 2} = \epsilon$。

4. **思考题**：为什么 DDPM 使用 $D_{KL}(q \| p)$ 而不是 $D_{KL}(p \| q)$？两种方向有什么不同的含义？（提示：考虑哪种方向可以让 $q$ 覆盖 $p$ 的所有模式）
