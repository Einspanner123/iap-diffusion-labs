# 变分推断与变分自编码器（VAE）系统性学习手册

> 本教程从贝叶斯推断出发，系统讲解变分推断的数学原理，并深入理解变分自编码器（VAE）的设计思想，最终建立与扩散模型的联系。

---

## 目录

1. [贝叶斯推断基础](#一贝叶斯推断基础)
2. [变分推断的动机](#二变分推断的动机)
3. [变分推断的数学原理](#三变分推断的数学原理)
4. [变分自编码器（VAE）](#四变分自编码器vae)
5. [重参数化技巧](#五重参数化技巧)
6. [解码器分布的选择](#六解码器分布的选择)
7. [后验坍塌问题](#七后验坍塌问题)
8. [β-VAE 与解耦表示](#八β-vae-与解耦表示)
9. [VQ-VAE：离散潜变量](#九vq-vae离散潜变量)
10. [VAE 与扩散模型的联系](#十vae-与扩散模型的联系)
11. [实践代码示例](#十一实践代码示例)
12. [进阶阅读](#十二进阶阅读)

---

## 一、贝叶斯推断基础

### 1.1 贝叶斯定理

**贝叶斯公式：**

$$p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$$

| 符号 | 名称 | 含义 |
|------|------|------|
| $p(\theta|D)$ | **后验分布** | 给定数据后，参数的概率分布 |
| $p(D|\theta)$ | **似然函数** | 参数为 $\theta$ 时，观测到数据的概率 |
| $p(\theta)$ | **先验分布** | 观测数据前，对参数的信念 |
| $p(D)$ | **证据/边缘似然** | 数据的边际概率（归一化常数） |

### 1.2 贝叶斯推断的目标

**目标**：计算后验分布 $p(\theta|D)$

**问题**：证据 $p(D)$ 通常难以计算！

$$p(D) = \int p(D|\theta)p(\theta) d\theta$$

这个积分在大多数情况下**不可解析计算**（intractable）。

### 1.3 例子：硬币投掷

假设我们投掷硬币 10 次，观察到 7 次正面。

**参数**：$\theta = P(\text{正面})$

**先验**：$\theta \sim \text{Beta}(2, 2)$（假设硬币大致公平）

**似然**：$p(D|\theta) = \theta^7 (1-\theta)^3$

**后验**：$p(\theta|D) \propto \theta^7 (1-\theta)^3 \cdot \theta^{2-1} (1-\theta)^{2-1} = \theta^8 (1-\theta)^4$

这个例子中后验有解析解（Beta 分布），但复杂模型中通常没有。

---

## 二、变分推断的动机

### 2.1 贝叶斯推断的困境

对于复杂模型（如神经网络）：

1. **后验分布 $p(\theta|D)$ 无法解析计算**
2. **证据 $p(D)$ 的积分无法求解**
3. **MCMC 方法计算代价太高**

### 2.2 解决思路：近似推断

既然无法精确计算后验，那就**找一个简单的分布来近似它**！

**核心思想**：
- 选择一个简单分布族 $q(\theta|\phi)$（如高斯分布）
- 找到最优参数 $\phi^*$，使得 $q(\theta|\phi^*)$ 最接近真实后验 $p(\theta|D)$

### 2.3 变分推断 vs MCMC

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **MCMC** | 从后验采样，用样本近似 | 渐近精确 | 收敛慢，计算贵 |
| **变分推断** | 优化简单分布来近似 | 快速，可扩展 | 是近似，有偏差 |

---

## 三、变分推断的数学原理

### 3.1 问题形式化

**目标**：找到 $q^*(\theta)$ 近似 $p(\theta|D)$

**分布族选择**：$q(\theta|\phi)$，其中 $\phi$ 是变分参数

### 3.2 KL 散度：衡量分布差异

**KL 散度（Kullback-Leibler Divergence）**：

$$\text{KL}(q(\theta) \| p(\theta)) = \int q(\theta) \log \frac{q(\theta)}{p(\theta)} d\theta$$

**性质**：
- $\text{KL}(q\|p) \geq 0$（非负性）
- $\text{KL}(q\|p) = 0 \iff q = p$（当且仅当两分布相等）
- 不对称：$\text{KL}(q\|p) \neq \text{KL}(p\|q)$

**KL 散度不对称的含义：**

| 方向 | 名称 | 特点 |
|------|------|------|
| $\text{KL}(q \mid p)$ | I-projection / 时刻匹配 | $q$ 会"回避" $p$ 为零的区域，$q$ 比 $p$ 更窄 |
| $\text{KL}(p \mid q)$ | M-projection / 均值匹配 | $q$ 会"覆盖" $p$ 的所有区域，$q$ 比 $p$ 更宽 |

> 💡 VAE 使用 $\text{KL}(q\|p)$ 方向，这意味着近似后验 $q(z|x)$ 会比真实后验更窄，倾向于"保守"地使用潜变量。

### 3.3 变分推断的目标函数

**优化目标**：最小化 KL 散度

$$\phi^* = \arg\min_\phi \text{KL}(q(\theta|\phi) \| p(\theta|D))$$

**展开 KL 散度**：

$$\begin{aligned}
\text{KL}(q(\theta|\phi) \| p(\theta|D)) 
&= \int q(\theta|\phi) \log \frac{q(\theta|\phi)}{p(\theta|D)} d\theta \\
&= \int q(\theta|\phi) \log q(\theta|\phi) d\theta - \int q(\theta|\phi) \log p(\theta|D) d\theta \\
&= \mathbb{E}_{q}[\log q(\theta|\phi)] - \mathbb{E}_{q}[\log p(\theta|D)]
\end{aligned}$$

### 3.4 证据下界（ELBO）的推导

**关键技巧**：引入联合分布 $p(D, \theta) = p(D|\theta)p(\theta)$

$$\begin{aligned}
\log p(D) &= \log \int p(D, \theta) d\theta \\
&= \log \int q(\theta|\phi) \frac{p(D, \theta)}{q(\theta|\phi)} d\theta \\
&\geq \int q(\theta|\phi) \log \frac{p(D, \theta)}{q(\theta|\phi)} d\theta \quad \text{(Jensen 不等式)} \\
&= \mathbb{E}_{q}[\log p(D, \theta) - \log q(\theta|\phi)] \\
&= \text{ELBO}(\phi)
\end{aligned}$$

**ELBO（Evidence Lower BOund）**：

$$\boxed{\text{ELBO}(\phi) = \mathbb{E}_{q(\theta|\phi)}[\log p(D|\theta) + \log p(\theta) - \log q(\theta|\phi)]}$$

### 3.5 ELBO 与 KL 散度的关系

$$\log p(D) = \text{ELBO}(\phi) + \text{KL}(q(\theta|\phi) \| p(\theta|D))$$

由于 $\log p(D)$ 是常数（与 $\phi$ 无关），**最大化 ELBO 等价于最小化 KL 散度**！

$$\max_\phi \text{ELBO}(\phi) \iff \min_\phi \text{KL}(q(\theta|\phi) \| p(\theta|D))$$

### 3.6 ELBO 的两项分解

$$\text{ELBO} = \underbrace{\mathbb{E}_{q}[\log p(D|\theta)]}_{\text{重构项}} - \underbrace{\text{KL}(q(\theta|\phi) \| p(\theta))}_{\text{先验约束}}$$

**直观理解**：
- **重构项**：近似后验下的期望似然，希望模型能解释数据
- **先验约束**：KL 散度惩罚，让近似后验不要偏离先验太远

**这两项的博弈：**

```
重构项: "让 q(z|x) 包含尽可能多的信息来重建 x"
         → q(z|x) 想偏离 p(z)，变得有结构

KL 项:  "让 q(z|x) 保持接近 p(z) = N(0,I)"
         → q(z|x) 想保持简单，接近标准正态

训练过程就是两者的平衡！
```

---

## 四、变分自编码器（VAE）

### 4.1 VAE 的动机

**问题**：传统自编码器只能学习确定性映射，无法生成新样本。

**VAE 的想法**：学习数据的**概率生成模型**！

```
生成过程：
z ~ p(z)           # 从先验采样
x ~ p(x|z)         # 从条件分布生成数据

推断过程：
z ~ q(z|x)         # 从近似后验推断潜变量
```

### 4.2 VAE 的概率图模型

| 分布 | 名称 | VAE 中的角色 |
|------|------|-------------|
| $p(z)$ | 先验分布 | 通常设为 $\mathcal{N}(0, I)$ |
| $p_\theta(x|z)$ | 似然/生成器 | 解码器（Decoder），参数为 $\theta$ |
| $q_\phi(z|x)$ | 近似后验 | 编码器（Encoder），参数为 $\phi$ |

### 4.3 VAE 的 ELBO

将变分推断应用到 VAE：

- 参数 $\theta$ → 潜变量 $z$
- 数据 $D$ → 单个样本 $x$
- 变分分布 $q(\theta|\phi)$ → $q_\phi(z|x)$（编码器）

**VAE 的 ELBO**：

$$\begin{aligned}
\text{ELBO} &= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)] \\
&= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重构项}} - \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 正则化}}
\end{aligned}$$

### 4.4 VAE 的损失函数

**训练目标**：最大化 ELBO（等价于最小化负 ELBO）

$$\boxed{\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x) \| p(z))}$$

| 项 | 作用 | 直观理解 |
|------|------|----------|
| **重构损失** $-\mathbb{E}_{q}[\log p_\theta(x \mid z)]$ | 让解码器能重建输入 | "解码器要能还原图片" |
| **KL 正则化** $\text{KL}(q_\phi(z \mid x) \mid p(z))$ | 让编码器输出接近先验 | "潜变量要保持标准正态分布" |

### 4.5 具体实现：高斯编码器

**编码器输出**：

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

- $\mu_\phi(x)$：编码器输出的均值向量，形状 $(d,)$
- $\sigma_\phi^2(x)$：编码器输出的方差向量，形状 $(d,)$（对角协方差）
- $d$：潜变量维度

> ⚠️ **注意对角协方差的假设**：$q_\phi(z|x)$ 的协方差矩阵是对角的，意味着潜变量的各维度相互独立。这是一个重要的简化假设，被称为 **mean-field approximation**。

### 4.6 KL 散度的解析解

对于两个高斯分布 $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ 和 $p(z) = \mathcal{N}(0, I)$：

$$\text{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2} \sum_{j=1}^{d} \left( \mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1 \right)$$

**逐项解释：**

| 项 | 含义 |
|------|------|
| $\mu_j^2$ | 惩罚均值偏离 0 |
| $\sigma_j^2$ | 惩罚方差偏离 1（过大） |
| $-\log(\sigma_j^2)$ | 惩罚方差过小（鼓励信息编码） |
| $-1$ | 常数，使得 $q = p$ 时 KL = 0 |

---

## 五、重参数化技巧

### 5.1 问题：如何对随机节点反向传播？

VAE 的计算图：

```
x → Encoder → μ, σ → 采样 z → Decoder → x̂
                    ↑
                随机性！
```

**问题**：采样操作 $z \sim \mathcal{N}(\mu, \sigma^2)$ 是随机的，无法直接反向传播！

### 5.2 重参数化技巧（Reparameterization Trick）

**核心思想**：将随机性转移到输入噪声上！

**标准采样**：
$$z \sim \mathcal{N}(\mu, \sigma^2)$$

**重参数化**：
$$\begin{aligned}
\epsilon &\sim \mathcal{N}(0, I) \\
z &= \mu + \sigma \odot \epsilon
\end{aligned}$$

其中 $\odot$ 表示逐元素乘法。

### 5.3 为什么有效？

**原始形式**（不可导）：
$$z \sim \mathcal{N}(\mu, \sigma^2) \quad \text{（采样是随机操作）}$$

**重参数化形式**（可导）：
$$\begin{aligned}
\epsilon &\sim \mathcal{N}(0, I) \quad \text{（固定分布，不需要对 } \epsilon \text{ 求梯度）} \\
z &= \mu + \sigma \odot \epsilon \quad \text{（确定性操作，可以对 } \mu, \sigma \text{ 反向传播）}
\end{aligned}$$

**梯度计算：**

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z}$$

$$\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \epsilon$$

### 5.4 计算图对比

**原始形式**：
```
x → Encoder → μ, σ → z ~ N(μ,σ²) → Decoder → x̂
                    ❌ 梯度在此中断
```

**重参数化**：
```
x → Encoder → μ, σ ──┐
                     (+) → z → Decoder → x̂
ε ~ N(0,I) → (×) ────┘
✅ 梯度可以流回 μ 和 σ
```

### 5.5 重参数化技巧的更一般形式

重参数化不限于高斯分布。对于任何可逆变换 $z = g(\epsilon, \phi)$，其中 $\epsilon$ 从固定分布采样：

| 分布 | 采样方法 | 重参数化 |
|------|----------|----------|
| 高斯 $\mathcal{N}(\mu, \sigma^2)$ | Box-Muller | $z = \mu + \sigma \cdot \epsilon$ |
| 指数 $\text{Exp}(\lambda)$ | 逆 CDF | $z = -\frac{1}{\lambda}\log(1-\epsilon)$ |
| Gumbel-Softmax | Gumbel-Max | $z = \text{onehot}(\arg\max(\log \pi + g))$ |

---

## 六、解码器分布的选择

### 6.1 解码器分布决定了重构损失的形式

**这是 VAE 实践中最容易被忽略的关键点！** 解码器 $p_\theta(x|z)$ 的分布假设直接决定了重构损失的形式。

### 6.2 伯努利解码器（Binary Data）

**适用场景**：二值数据（如二值化的 MNIST）

$$p_\theta(x|z) = \prod_{i=1}^{D} \text{Bernoulli}(x_i; \hat{x}_i(z))$$

其中 $\hat{x}_i(z) \in [0, 1]$ 是解码器输出的第 $i$ 个像素值。

**重构损失**：

$$-\log p_\theta(x|z) = -\sum_{i=1}^{D} \left[ x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i) \right]$$

**这就是二元交叉熵（BCE）！**

**实现**：解码器最后一层用 Sigmoid 激活 + `F.binary_cross_entropy`

### 6.3 高斯解码器（Continuous Data）

**适用场景**：连续值数据（如自然图像的像素值）

$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$$

**情况 A：固定方差 $\sigma^2$**

$$-\log p_\theta(x|z) = \frac{1}{2\sigma^2}\|x - \mu_\theta(z)\|^2 + \frac{D}{2}\log(2\pi\sigma^2)$$

**重构损失 = MSE（均方误差）× $\frac{1}{2\sigma^2}$ + 常数**

> ⚠️ 当 $\sigma^2 = 1$ 时，$-\log p_\theta(x|z) \propto \|x - \mu_\theta(z)\|^2$，即 MSE 损失。

**情况 B：学习的方差**

解码器同时输出 $\mu_\theta(z)$ 和 $\log \sigma_\theta^2(z)$：

$$-\log p_\theta(x|z) = \frac{1}{2}\sum_{i=1}^{D}\left[\frac{(x_i - \mu_i)^2}{\sigma_i^2} + \log \sigma_i^2\right] + C$$

### 6.4 拉普拉斯解码器

$$p_\theta(x|z) = \prod_{i=1}^{D} \text{Laplace}(x_i; \mu_i(z), b)$$

**重构损失 = L1 损失（绝对误差）**

$$-\log p_\theta(x|z) = \frac{1}{b}\|x - \mu_\theta(z)\|_1 + C$$

### 6.5 解码器选择总结

| 解码器分布 | 适用数据 | 重构损失 | 解码器输出层 |
|-----------|----------|----------|-------------|
| Bernoulli | 二值 | BCE | Sigmoid |
| Gaussian（固定 $\sigma^2$） | 连续 | MSE | 线性（无激活） |
| Gaussian（学习 $\sigma^2$） | 连续 | 加权 MSE + $\log\sigma^2$ | 线性 + 额外输出 |
| Laplace | 连续（稀疏误差） | L1 | 线性 |

> ⚠️ **常见错误**：对连续数据使用 Sigmoid + BCE。这假设数据是伯努利的，但自然图像的像素值是连续的。虽然实践中"也能用"，但理论上不正确，可能导致生成质量下降。

### 6.6 确定性解码器？

有时会看到 $p(x|z) = \delta(x - \mu_\theta(z))$ 的说法。**这严格来说不是一个合法的概率分布**（delta 函数不是密度函数），但可以理解为高斯解码器 $\sigma^2 \to 0$ 的极限情况。在实践中，我们总是使用有限方差的分布。

---

## 七、后验坍塌问题

### 7.1 什么是后验坍塌？

**后验坍塌（Posterior Collapse）** 是 VAE 训练中最常见也最棘手的问题。

**定义**：编码器学到的近似后验 $q_\phi(z|x)$ 退化为先验 $p(z)$，即：

$$q_\phi(z|x) \approx p(z) = \mathcal{N}(0, I) \quad \text{对所有 } x$$

**表现**：
- 编码器输出的 $\mu_\phi(x) \approx 0$，$\sigma_\phi^2(x) \approx 1$
- 潜变量 $z$ 不包含任何关于 $x$ 的信息
- 解码器完全忽略 $z$，变成一个无条件模型
- 生成的样本缺乏多样性

### 7.2 为什么会发生后验坍塌？

**根本原因：** KL 散度的惩罚过强，编码器发现"不编码信息"比"编码信息"更划算。

```
训练早期：
  重构项梯度大 → 编码器开始编码信息 → KL 项增大
  ↓
训练后期：
  KL 项梯度大 → 编码器放弃编码信息 → 回退到先验
  ↓
结果：
  q(z|x) ≈ p(z)，z 不包含信息
```

**数学解释：** 当解码器 $p_\theta(x|z)$ 足够强大时（如自回归解码器），即使 $z$ 不包含信息，解码器也能生成合理的数据。此时 KL 项的梯度主导了训练，推动 $q_\phi(z|x)$ 趋向 $p(z)$。

### 7.3 后验坍塌的检测

| 指标 | 正常训练 | 后验坍塌 |
|------|----------|----------|
| KL 散度 | 适中（如 10-100） | 接近 0 |
| $\mu_\phi(x)$ 的方差 | 较大 | 接近 0 |
| $\sigma_\phi^2(x)$ | 适中 | 接近 1 |
| 活跃潜变量维度 | 接近 $d$ | 远小于 $d$ |
| 生成多样性 | 多样 | 单调 |

### 7.4 后验坍塌的解决方案

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **KL 退火（KL Annealing）** | 训练初期 KL 权重从 0 线性增加到 1 | 简单有效，但需要调调度 |
| **自由比特（Free Bits）** | 设 KL 下限：每维最小 $\lambda$ nats | 保证每维至少编码 $\lambda$ nats 信息 |
| **弱化解码器** | 使用较弱的解码器（如非自回归） | 简单但限制模型能力 |
| **$\delta$-VAE** | 修改先验为更灵活的分布 | 理论优美但实现复杂 |
| **周期性 KL 调度** | KL 权重周期性变化 | 实践中有效 |

**KL 退火的具体实现：**

$$\mathcal{L} = -\mathbb{E}_{q}[\log p_\theta(x|z)] + \beta(t) \cdot \text{KL}(q_\phi(z|x) \| p(z))$$

其中 $\beta(t)$ 从 0 线性增加到 1：

$$\beta(t) = \min\left(\frac{t}{T_{\text{warmup}}}, 1\right)$$

**自由比特的具体实现：**

$$\text{KL}_{\text{fb},j} = \max(\lambda, \text{KL}_j)$$

其中 $\text{KL}_j$ 是第 $j$ 维的 KL 散度，$\lambda$ 是阈值（通常 0.1-0.5 nats）。

---

## 八、β-VAE 与解耦表示

### 8.1 β-VAE 的定义

**β-VAE**（Higgins et al., 2017）通过引入超参数 $\beta$ 来控制 KL 散度的权重：

$$\boxed{\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot \text{KL}(q_\phi(z|x) \| p(z))}$$

| $\beta$ 值 | 效果 |
|------------|------|
| $\beta = 1$ | 标准 VAE |
| $\beta > 1$ | 更强的 KL 惩罚 → 更解耦的表示，但重构质量下降 |
| $\beta < 1$ | 更弱 KL 惩罚 → 更好的重构，但表示不解耦 |

### 8.2 什么是解耦表示？

**解耦表示（Disentangled Representation）**：潜变量的每个维度独立控制数据的某个语义因子。

```
解耦的例子（人脸生成）：
z₁ → 控制朝向（左/右）
z₂ → 控制表情（笑/不笑）
z₃ → 控制光照（明/暗）
...

改变 z₁ 只改变朝向，不影响其他属性！
```

### 8.3 β-VAE 的直觉

**增大 $\beta$ 的效果：**

```
β = 1 (标准 VAE):
  编码器可以"随意"使用潜变量空间
  → 信息压缩不够，表示纠缠

β > 1:
  KL 惩罚更强，编码器被迫更高效地使用潜变量
  → 每个维度必须独立编码最显著的特征
  → 表示更解耦

β 过大:
  KL 惩罚太强，编码器无法编码足够信息
  → 后验坍塌！
```

### 8.4 β-VAE 与信息瓶颈

β-VAE 可以从**信息瓶颈（Information Bottleneck）**的角度理解：

$$\min I(z; x) \quad \text{s.t.} \quad I(z; x) \geq I_{\text{target}}$$

- $\beta$ 控制了信息压缩的程度
- $\beta$ 越大，$z$ 中保留的关于 $x$ 的信息越少
- 但保留的信息更加"精炼"（解耦）

---

## 九、VQ-VAE：离散潜变量

### 9.1 为什么需要离散潜变量？

标准 VAE 的潜变量是连续的 $z \in \mathbb{R}^d$，但很多场景需要离散表示：

- 语言是离散的（词/标记）
- 音频可以用离散码本表示
- 离散表示更易于解释和操作

### 9.2 VQ-VAE 的核心思想

**Vector Quantized VAE**（van den Oord et al., 2017）用**码本（Codebook）**替代连续潜变量：

$$\text{Codebook: } E = \{e_1, e_2, \ldots, e_K\}, \quad e_k \in \mathbb{R}^d$$

**量化过程**：

$$z_q(x) = e_k, \quad k = \arg\min_j \|z_e(x) - e_j\|$$

其中 $z_e(x)$ 是编码器输出的连续向量，$z_q(x)$ 是量化后的离散码。

### 9.3 VQ-VAE 的损失函数

$$\mathcal{L} = \underbrace{\|x - D(z_q)\|^2}_{\text{重构}} + \underbrace{\|sg[z_e] - e_k\|^2}_{\text{码本更新}} + \underbrace{\beta\|z_e - sg[e_k]\|^2}_{\text{承诺损失}}$$

其中 $sg[\cdot]$ 是 stop-gradient 操作。

| 项 | 作用 |
|------|------|
| 重构损失 | 让解码器还原输入 |
| 码本更新 | 让码本向量 $e_k$ 靠近编码器输出 |
| 承诺损失 | 让编码器输出靠近选中的码本向量 |

### 9.4 VQ-VAE 与标准 VAE 的对比

| 特性 | VAE | VQ-VAE |
|------|-----|--------|
| 潜变量 | 连续 $\mathcal{N}(\mu, \sigma^2)$ | 离散（码本索引） |
| 先验 | $\mathcal{N}(0, I)$ | 分类分布（训练后学习） |
| KL 散度 | 有（解析计算） | 无（用承诺损失替代） |
| 后验坍塌 | 有此问题 | 无此问题 |
| 生成模型 | 先验采样 → 解码 | PixelCNN 采样 → 解码 |

### 9.5 VQ-VAE 与扩散模型的联系

**VQ-VAE-2**（Razavi et al., 2019）将 VQ-VAE 与层次化自回归先验结合，是扩散模型之前最强大的生成模型之一。

**Latent Diffusion**（Rombach et al., 2022，即 Stable Diffusion）的核心思想是先把图像压到潜空间，再在潜空间里做扩散；它和 VQ-VAE 一样都强调“先学一个表示，再在表示空间里生成”，但具体实现并不是典型的 VQ-VAE 路线。

---

## 十、VAE 与扩散模型的联系

### 10.1 层次化 VAE 视角

扩散模型可以看作**层次化 VAE**的特殊形式：

**VAE**（单层）：
$$\begin{aligned}
\text{先验：} & z \sim p(z) \\
\text{生成：} & x \sim p(x|z) \\
\text{推断：} & z \sim q(z|x)
\end{aligned}$$

**扩散模型**（多层/时序）：
$$\begin{aligned}
\text{先验：} & x_T \sim p(x_T) = \mathcal{N}(0, I) \\
\text{生成：} & x_{t-1} \sim p_\theta(x_{t-1}|x_t), \quad t=T,\ldots,1 \\
\text{推断：} & x_t \sim q(x_t|x_{t-1}), \quad t=1,\ldots,T
\end{aligned}$$

### 10.2 ELBO 的对比

**VAE 的 ELBO**：
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$$

**扩散模型的 ELBO**：
$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\sum_{t=1}^T \log p_\theta(x_{t-1}|x_t)\right] - \text{KL}(q(x_T|x_0) \| p(x_T))$$

### 10.3 关键区别

| 特性 | VAE | 扩散模型 |
|------|-----|---------|
| **潜变量** | 单层 $z$ | 多层 $\{x_1, \ldots, x_T\}$ |
| **先验** | $p(z) = \mathcal{N}(0, I)$ | $p(x_T) = \mathcal{N}(0, I)$ |
| **生成过程** | 一步：$x \sim p(x \mid z)$ | 多步：$x_0 \sim p(x_0 \mid x_1) \cdots p(x_{T-1} \mid x_T)$ |
| **推断过程** | 一步：$z \sim q(z \mid x)$ | 多步：$x_T \sim q(x_T \mid x_{T-1}) \cdots q(x_1 \mid x_0)$ |
| **推断是否学习** | 是（编码器） | **否（固定的加噪过程）** |
| **后验坍塌** | 有此问题 | 无此问题 |

### 10.4 扩散模型作为 VAE 的特例

DDPM 可以看作**固定推断过程**的层次化 VAE：

1. **推断过程 $q$ 是固定的**（不是学习的）
   - $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$

2. **只学习生成过程 $p_\theta$**
   - $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

3. **ELBO 简化为噪声预测损失**
   - 经过推导，ELBO 简化为 $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

### 10.5 为什么扩散模型没有后验坍塌？

**关键区别**：扩散模型的推断过程是**固定的**，不需要学习。

| VAE | 扩散模型 |
|-----|---------|
| 编码器 $q_\phi(z \mid x)$ 需要学习 | 加噪过程 $q(x_t \mid x_{t-1})$ 是固定的 |
| 编码器可能"偷懒"不编码信息 | 加噪过程一定会添加噪声 |
| KL 惩罚可能导致后验坍塌 | 没有 KL 惩罚（推断是固定的） |

### 10.6 从 VAE 理解 DDPM 的训练目标

**VAE 视角**：
- 编码器 $q$：固定为加噪过程
- 解码器 $p_\theta$：学习去噪过程
- ELBO 最大化 → 噪声预测 MSE 最小化

**关键洞察**：DDPM 的神经网络 $\epsilon_\theta$ 实际上是在学习**解码器**（从噪声恢复数据）！

---

## 十一、实践代码示例

### 11.1 完整 VAE 训练代码（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, decoder_type='bernoulli'):
        super().__init__()
        self.decoder_type = decoder_type
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if decoder_type == 'bernoulli':
            self.fc_out = nn.Linear(hidden_dim, input_dim)
        elif decoder_type == 'gaussian':
            self.fc_mu_out = nn.Linear(hidden_dim, input_dim)
            self.fc_logvar_out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        if self.decoder_type == 'bernoulli':
            return torch.sigmoid(self.fc_out(h))
        elif self.decoder_type == 'gaussian':
            mu = self.fc_mu_out(h)
            logvar = self.fc_logvar_out(h)
            return mu, logvar

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        if self.decoder_type == 'bernoulli':
            x_recon = self.decode(z)
            return x_recon, mu, logvar
        elif self.decoder_type == 'gaussian':
            mu_out, logvar_out = self.decode(z)
            return mu_out, logvar_out, mu, logvar


def vae_loss_bernoulli(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def vae_loss_gaussian(mu_out, logvar_out, x, mu, logvar, beta=1.0):
    recon_loss = 0.5 * torch.sum(
        logvar_out + (x - mu_out).pow(2) / logvar_out.exp()
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


class KLAnnealer:
    def __init__(self, total_steps, warmup_ratio=0.3):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = 0

    def get_beta(self):
        if self.current_step < self.warmup_steps:
            return self.current_step / self.warmup_steps
        return 1.0

    def step(self):
        self.current_step += 1


def train_vae(model, dataloader, num_epochs=10, lr=1e-3, use_kl_annealing=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = num_epochs * len(dataloader)
    annealer = KLAnnealer(total_steps) if use_kl_annealing else None

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.view(-1, 784)

            optimizer.zero_grad()

            if model.decoder_type == 'bernoulli':
                x_recon, mu, logvar = model(x)
                beta = annealer.get_beta() if annealer else 1.0
                loss = vae_loss_bernoulli(x_recon, x, mu, logvar, beta=beta)
            else:
                mu_out, logvar_out, mu, logvar = model(x)
                beta = annealer.get_beta() if annealer else 1.0
                loss = vae_loss_gaussian(mu_out, logvar_out, x, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            if annealer:
                annealer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')


def sample_from_vae(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        if model.decoder_type == 'bernoulli':
            x_gen = model.decode(z)
        else:
            mu_out, _ = model.decode(z)
            x_gen = mu_out
    return x_gen


if __name__ == '__main__':
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = VAE(latent_dim=20, decoder_type='bernoulli')
    train_vae(model, train_loader, num_epochs=10)

    samples = sample_from_vae(model, num_samples=10)
```

---

## 十二、进阶阅读

### 12.1 经典论文

1. **VAE 开山之作**：
   - Auto-Encoding Variational Bayes (Kingma & Welling, 2014)

2. **β-VAE**：
   - β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (Higgins et al., 2017)

3. **VQ-VAE**：
   - Neural Discrete Representation Learning (van den Oord et al., 2017)

4. **扩散模型奠基**：
   - Denoising Diffusion Probabilistic Models (Ho et al., 2020)

5. **变分推断综述**：
   - Variational Inference: A Review for Statisticians (Blei et al., 2017)

### 12.2 深入主题

1. **改进的 VAE**：
   - β-VAE（解耦表示学习）
   - VQ-VAE / VQ-VAE-2（离散潜变量）
   - Hierarchical VAE（层次化结构）
   - NVAE（深度层次化 VAE）

2. **后验坍塌相关**：
   - KL Annealing (Bowman et al., 2016)
   - Free Bits (Kingma et al., 2017)
   - δ-VAE (Razavi et al., 2019)

3. **扩散模型进阶**：
   - DDIM（确定性采样）
   - Score-based Models（分数匹配视角）
   - Latent Diffusion（潜空间扩散）

4. **变分推断扩展**：
   - Importance Weighted Autoencoders (IWAE)
   - Normalizing Flows for Variational Inference
   - Stein Variational Gradient Descent

### 12.3 学习路径建议

```
入门 → 贝叶斯统计 → 变分推断 → VAE → 扩散模型
        ↓           ↓         ↓       ↓
    概率基础   近似推断   深度生成   前沿研究
```

---

## 附录：关键公式速查表

### A.1 贝叶斯定理

$$p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$$

### A.2 KL 散度

$$\text{KL}(q\|p) = \mathbb{E}_q\left[\log\frac{q}{p}\right]$$

### A.3 ELBO

$$\text{ELBO} = \mathbb{E}_q[\log p(D|\theta)] - \text{KL}(q(\theta)\|p(\theta))$$

### A.4 VAE 损失

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x)\|p(z))$$

### A.5 β-VAE 损失

$$\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \beta \cdot \text{KL}(q_\phi(z|x)\|p(z))$$

### A.6 重参数化

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### A.7 高斯 KL 散度（解析解）

$$\text{KL}(\mathcal{N}(\mu,\text{diag}(\sigma^2))\|\mathcal{N}(0,I)) = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

### A.8 解码器分布与重构损失对应表

| 解码器 | 重构损失 |
|--------|----------|
| Bernoulli | BCE |
| Gaussian（固定 $\sigma^2$） | $\frac{1}{2\sigma^2}\Vert x-\hat{x}\Vert^2$ |
| Gaussian（学习 $\sigma^2$） | $\frac{1}{2}\sum\left[\frac{(x_i-\hat{x}_i)^2}{\sigma_i^2}+\log\sigma_i^2\right]$ |
| Laplace | $\frac{1}{b}\Vert x-\hat{x}\Vert_1$ |

---

## 总结

变分推断的核心思想：

1. **问题**：后验分布难以精确计算
2. **思路**：用简单分布近似复杂分布
3. **方法**：最大化 ELBO（证据下界）
4. **应用**：VAE、扩散模型等深度生成模型

VAE 的关键设计选择：

1. **编码器分布**：高斯（对角协方差），用重参数化技巧训练
2. **解码器分布**：根据数据类型选择（Bernoulli/Gaussian/Laplace）
3. **KL 权重**：$\beta=1$ 是标准 VAE，$\beta>1$ 促进解耦，$\beta$ 退火防止后验坍塌

VAE 与扩散模型的统一视角：

- 都是**变分自编码器**的不同形式
- 都使用**重参数化技巧**实现可导采样
- 都通过**最大化 ELBO**训练
- 区别在于**潜变量结构**和**生成过程**
- 扩散模型通过固定推断过程避免了后验坍塌

掌握变分推断，就掌握了理解现代生成模型的钥匙！
