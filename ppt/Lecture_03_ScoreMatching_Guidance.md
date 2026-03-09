# 📚 MIT IAP 2026 · Lecture 3: Score Matching and Guidance

## 分数匹配与引导 —— 面向统计与概率论初学者

---

## 一、本讲三大主题

1. **Score Matching（分数匹配）**：另一种训练 Flow/Diffusion 模型的方法
2. **Stochastic Sampling（随机采样）**：用 SDE 代替 ODE 生成样本
3. **Classifier-Free Guidance（无分类器引导）**：如何让生成结果听从 prompt

### 1.1 本讲主线

这一讲其实做了两次“改写”：

1. 把“预测向量场”改写成“预测 score”
2. 把“普通条件生成”改写成“带 guidance 的条件生成”

前者回答的是“训练目标能不能换一种表达”，后者回答的是“采样时怎样让模型更听 prompt”。

---

## 二、复习：Lecture 2 的核心结果

### Flow Matching 训练算法（Algorithm 3）

```
对每个 mini-batch:
    1. 采样数据 z，随机时间 t，采样 x ~ p_t(·|z)
    2. 计算损失 L(θ) = ‖u_t^θ(x) - u_t^target(x|z)‖²
    3. 梯度下降更新 θ
```

> 关键思想：通过拟合许多不同数据点 $z$ 的条件向量场，神经网络学会了边际向量场。

---

## 三、Score Matching——另一个视角

### 3.1 什么是 Score Function（分数函数）？

$$\boxed{\text{Score Function} = \nabla_x \log q(x)}$$

**逐项解读：**

| 符号 | 含义 |
|------|------|
| $q(x)$ | 某个概率密度函数 |
| $\log q(x)$ | **对数似然**（log-likelihood） |
| $\nabla_x \log q(x)$ | 对数似然关于 $x$ 的**梯度**（一个向量） |

**直觉：** Score function 是一个**向量场**，在每个点 $x$ 指向"概率密度增长最快的方向"。

> 🎯 **生活比喻：** 如果概率密度是一座山的高度，score function 就是每个位置的"上山方向"——指向山顶（高概率区域）。

### 3.2 高斯条件概率路径的 Score

对于 $p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，其 score 为：

$$\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}$$

**推导过程（初学者友好版）：**

1. 写出高斯密度：

$$p_t(x|z) = \frac{1}{(2\pi)^{d/2}\beta_t^d}\exp\left(-\frac{1}{2\beta_t^2}\|x - \alpha_t z\|^2\right)$$

2. 取对数：

$$\log p_t(x|z) = -\frac{d}{2}\log(2\pi) - d\log\beta_t - \frac{1}{2\beta_t^2}\|x - \alpha_t z\|^2$$

3. 对 $x$ 求梯度（前两项不含 $x$，梯度为零）：

$$\nabla_x \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}$$

> 📐 **用到的知识：** $\nabla_x \|x - c\|^2 = 2(x - c)$，这是向量微积分的基础公式。

**直觉：** Score 指向从当前位置 $x$ 到"目标中心" $\alpha_t z$ 的方向，且离中心越远，推力越大。就像弹簧——偏离越远，回复力越强。

### 3.3 条件向量场 vs 条件 Score：线性关系！

对于高斯概率路径，两者的公式：

| | 公式 | $z$ 前系数 | $x$ 前系数 |
|---|---|---|---|
| **条件向量场** $u_t^{\text{target}}(x\|z)$ | $\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}x$ | $\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t$ | $\frac{\dot{\beta}_t}{\beta_t}$ |
| **条件 Score** $\nabla \log p_t(x\|z)$ | $\frac{\alpha_t}{\beta_t^2}z - \frac{1}{\beta_t^2}x$ | $\frac{\alpha_t}{\beta_t^2}$ | $-\frac{1}{\beta_t^2}$ |

> 🔑 **关键观察：两者都是关于 $z$ 和 $x$ 的线性函数！** 只是系数不同。

### 3.4 重参数化公式：向量场 ↔ Score

定义两个时间相关的系数：

$$a_t = \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t, \qquad b_t = \frac{\dot{\alpha}_t}{\alpha_t}$$

则向量场可以用 score 来表达：

$$u_t^{\text{target}}(x|z) = a_t \nabla \log p_t(x|z) + b_t x$$

$$u_t^{\text{target}}(x) = a_t \nabla \log p_t(x) + b_t x$$

> 💡 **意义：** 学向量场和学 score 是**等价的**！早期的 Diffusion Model 先学 score，再转换成向量场。现在的 Flow Matching 直接学向量场。

**这一步为什么重要？**

因为 score 的统计意义非常直接：

- 它只关心“高概率方向在哪里”
- 不需要显式计算归一化常数
- 在扩散文献里，它天然对应“去噪”

所以许多早期扩散模型虽然表面上写的是 score matching，本质上仍是在学习生成过程的动力学。

### 3.5 Score Matching 训练算法（Algorithm 6）

```
对每个 mini-batch:
    1. 采样数据 z, 随机时间 t, 采样 x ~ p_t(·|z)
    2. 计算损失 L(θ) = ‖s_t^θ(x) - ∇ log p_t(x|z)‖²
    3. 梯度下降更新 θ
```

> 和 Flow Matching 几乎一模一样！只是把"预测向量场"换成了"预测 score"。

### 3.6 Denoising Score Matching——去噪视角

代入高斯路径的 score $\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}$，并用重参数化 $x = \alpha_t z + \beta_t \epsilon$：

$$\mathcal{L}_{\text{dsm}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\left\|s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t}\right\|^2\right]$$

**解读：** 网络需要预测**用来破坏数据的噪声** $\epsilon$（除以 $\beta_t$）！

> 🎯 这就是为什么叫 **Denoising（去噪）** Diffusion Model——网络本质上是在做"去噪"任务：给一个被噪声污染的样本，猜出噪声是什么。

**注意：** 当 $\beta_t$ 接近 0 时，$\frac{\epsilon}{\beta_t}$ 会变得非常大，导致**数值不稳定**。这是实践中需要注意的问题。

---

## 四、Fokker-Planck 方程与随机采样

### 4.1 Fokker-Planck 方程

回忆 SDE：$\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t$

如果 $X_t \sim p_t$，则 $p_t$ 满足 **Fokker-Planck 方程**：

$$\frac{\mathrm{d}}{\mathrm{d}t}p_t(x) = -\text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2}\Delta p_t(x)$$

**对比连续性方程（Lecture 2）：**

| | 连续性方程 (ODE) | Fokker-Planck 方程 (SDE) |
|---|---|---|
| 公式 | $\frac{\mathrm{d}}{\mathrm{d}t}p_t = -\text{div}(p_t u_t)$ | $\frac{\mathrm{d}}{\mathrm{d}t}p_t = -\text{div}(p_t u_t) + \frac{\sigma_t^2}{2}\Delta p_t$ |
| 多出来的项 | 无 | $\frac{\sigma_t^2}{2}\Delta p_t$（**热扩散项**） |

**各项含义：**
- $-\text{div}(p_t u_t)$：概率质量随向量场的**流动**（和 ODE 一样）
- $\frac{\sigma_t^2}{2}\Delta p_t$：概率质量的**热扩散/弥散**（像热量从高温区向低温区扩散）

> $\Delta$ 是拉普拉斯算子（所有二阶偏导数之和），控制"概率向周围扩散"的速率。

### 4.2 SDE Extension Trick——随机采样的核心

**核心定理：** 如果 ODE 的向量场 $u_t^{\text{target}}(x)$ 使得 $X_t \sim p_t$，那么以下 SDE **也能**使 $X_t \sim p_t$（对任意 $\sigma_t > 0$）：

$$\mathrm{d}X_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\right]\mathrm{d}t + \sigma_t \mathrm{d}W_t$$

**直觉：** 我们添加了随机噪声（$\sigma_t \mathrm{d}W_t$），但同时用 score function（$\frac{\sigma_t^2}{2}\nabla \log p_t$）做修正，两者刚好抵消，使得概率路径不变！

> 就像在河流里加了湍流（随机波动），但同时调整了河道形状，让水流的整体分布保持一样。

**代入神经网络：**

$$\mathrm{d}X_t = \left[\left(a_t + \frac{\sigma_t^2}{2}\right)s_t^\theta(X_t) + b_t X_t\right]\mathrm{d}t + \sigma_t \mathrm{d}W_t$$

### 4.3 为什么要用随机采样？

| 理论上 | 实践中 |
|--------|--------|
| 任何 $\sigma_t$ 都能正确采样 | 神经网络不完美（训练误差）|
| ODE 和 SDE 结果一样 | ODE 模拟有离散化误差 |

> **好消息：ODE 采样通常就够好了。** SDE 采样是可选的加分项，在某些下游任务（如蛋白质生成、微调）中有帮助。

### 4.4 附注：Langevin Dynamics

当向量场为零、概率路径恒定时，SDE 退化为 **Langevin dynamics**：

$$\mathrm{d}X_t = \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t$$

如果 $p_t = p_{\text{Boltzmann}} = \frac{1}{Z}\exp(-U(x))$（玻尔兹曼分布），这正是**分子动力学模拟**的基础。

> 粒子沿着势能下降的方向走（$\nabla \log p$ 项），同时受热运动的随机扰动（$\sigma_t \mathrm{d}W_t$ 项），最终达到热平衡。

---

## 五、Classifier-Free Guidance (CFG)——无分类器引导

### 5.1 问题背景：为什么需要引导？

**无引导生成（Unguided）：** "生成一张图片" → 结果随机，可能生成任何东西

**有引导生成（Guided）：** "生成一张猫在烤蛋糕的图片" → 需要结果符合 prompt

**朴素方法：** 直接训练条件向量场 $u_t^\theta(x|y)$，用它来采样。

**问题：** 朴素条件生成的效果**很差**！生成的图片不够忠实于 prompt，且质量不高。

### 5.2 Classifier Guidance（分类器引导）的思想

条件向量场可以分解为：

$$u_t^{\text{target}}(x|y) = \underbrace{u_t^{\text{target}}(x)}_{\text{无条件向量场}} + \underbrace{a_t \nabla_x \log p_t(y|x)}_{\text{prompt 相关的修正}}$$

**向量分解直觉：**
- $u_t^{\text{target}}(x)$：**基础方向**（不管 prompt，往"好图片"方向走）
- $a_t \nabla_x \log p_t(y|x)$：**修正方向**（往"更符合 prompt $y$"的方向偏）

**增强 prompt 效果：** 把修正项**放大** $w$ 倍：

$$\tilde{u}_t^w(x|y) = u_t^{\text{target}}(x) + w \cdot a_t \nabla_x \log p_t(y|x)$$

> $w > 1$ 时，模型更"听话"，生成结果更忠实于 prompt。

### 5.3 Classifier-Free Guidance（无分类器引导）

**问题：** $\nabla_x \log p_t(y|x)$ 需要一个额外的分类器，不方便。

**解决：** 注意到 prompt 修正项 = 条件向量场 − 无条件向量场：

$$u_t^{\text{target}}(x|y) - u_t^{\text{target}}(x) = a_t \nabla_x \log p_t(y|x)$$

所以 CFG 的公式变成（完全不需要分类器！）：

$$\boxed{\tilde{u}_t^w(x|y) = u_t^{\text{target}}(x) + w\Big(u_t^{\text{target}}(x|y) - u_t^{\text{target}}(x)\Big)}$$

等价形式：

$$\boxed{u_t^{\theta,w}(x) = (1 - w)\,u_t^\theta(x|\varnothing) + w\,u_t^\theta(x|y)}$$

其中 $\varnothing$ 表示"空标签"（无条件）。

**用大白话说：**
- $w = 1$：普通条件生成（不增强）
- $w > 1$：增强 prompt 的影响力（$w = 4.0$ 是常见选择）
- $w = 0$：完全无条件生成

### 5.4 CFG 训练算法（Algorithm 5）

```
输入: 配对数据集 (z, y) ~ p_data, 神经网络 u_t^θ
对每个 mini-batch:
    1. 采样数据 (z, y), 随机时间 t, 噪声 ε
    2. 构造 x = α_t z + β_t ε
    3. 以概率 p 丢弃标签: y ← ∅     ← 关键：随机丢标签！
    4. 计算损失 L(θ) = ‖u_t^θ(x|y) - u_t^target(x|z)‖²
    5. 梯度下降更新 θ
```

> 🔑 **核心技巧：** 训练时随机以概率 $p$ 把标签替换为空标签 $\varnothing$。这样**同一个网络**既学了条件生成（有标签时），也学了无条件生成（空标签时）。

### 5.5 CFG 采样算法（Algorithm 8）

```
输入: 训练好的 u_t^θ(x|y), prompt y, guidance scale w > 1
1. 初始化 X_0 ~ p_init
2. 模拟 ODE:
    dX_t = [(1-w)·u_t^θ(X_t|∅) + w·u_t^θ(X_t|y)] dt
3. 返回 X_1
```

### 5.6 CFG 的效果——惊人的质量提升

| $w = 1.0$（无引导） | $w = 4.0$（强引导） |
|---|---|
| 图片模糊、不贴合 prompt | 图片清晰、高度贴合 prompt |

> **Stable Diffusion 3 默认使用 $w \approx 4.0$。** 几乎所有你见到的 AI 生成图片/视频都使用了 CFG！

### 5.7 CFG 的理论代价

> ⚠️ **CFG 不再忠实于数据分布！** 当 $w > 1$ 时，生成的分布比真实数据分布更"尖锐"（集中在高概率区域），牺牲了多样性来换取质量。

这在实践中是值得的——用户通常更在意图片质量而非多样性。

---

## 六、扩散文献导读与术语对照

### 6.1 三种时间约定

| 约定 | 数据端 | 噪声端 | 代表文献 |
|------|--------|--------|---------|
| **Flow 约定** | $t = 1$ | $t = 0$ | Flow Matching, Rectified Flows, Stochastic Interpolants |
| **Diffusion 约定** | $t = 0$ | $t \to \infty$ | Score-based Diffusion Models with SDEs |
| **离散时间** | — | — | DDPM, DDIM |

### 6.2 三种等价的"加噪"描述

| 方法 | 公式 | 代表 |
|------|------|------|
| **概率路径** | $p_t(x\|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ | Flow Matching, Rectified Flows |
| **插值函数** | $I_t(\epsilon, z) = \alpha_t \epsilon + \beta_t z$ | Stochastic Interpolants |
| **前向 SDE** | $\mathrm{d}X_t = a_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t$ | Denoising Diffusion Models |

> 对于高斯概率路径，这三种描述是**完全等价的**！只是不同论文的"语言"不同。

### 6.3 前向 SDE 与反向 SDE

**前向 SDE（data → noise）：** 逐步给数据加噪

$$\mathrm{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\mathrm{d}t + g(t)\mathrm{d}\mathbf{w}$$

**反向 SDE（noise → data）：** 利用 score function 去噪

$$\mathrm{d}\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_x \log p_t(\mathbf{x})\right]\mathrm{d}t + g(t)\mathrm{d}\bar{\mathbf{w}}$$

> 反向 SDE 其实就是 SDE Extension Trick 的一个特例。

---

## 七、总结

| 概念 | 核心公式 | 直觉 |
|------|---------|------|
| **Score Function** | $\nabla \log p_t(x)$ | 指向高概率区域的"上山方向" |
| **Score Matching** | $\mathcal{L} = \|s_t^\theta(x) - \nabla \log p_t(x\|z)\|^2$ | 让网络学会预测"上山方向" |
| **Denoising** | 网络预测噪声 $\epsilon / \beta_t$ | 本质是"去噪"任务 |
| **向量场 ↔ Score** | $u_t = a_t \nabla \log p_t + b_t x$ | 两者等价，可互相转换 |
| **Fokker-Planck** | $\frac{\mathrm{d}}{\mathrm{d}t}p_t = -\text{div}(p_t u_t) + \frac{\sigma_t^2}{2}\Delta p_t$ | SDE 版的概率守恒 |
| **SDE Extension** | 加噪声 $\sigma_t \mathrm{d}W_t$ + score 修正 | 随机采样，概率路径不变 |
| **CFG** | $(1-w)u_t^\theta(x\|\varnothing) + wu_t^\theta(x\|y)$ | 放大 prompt 的影响力 |

> 🏆 **CFG 是实践中最重要的技术之一：** 没有它，AI 生成的图片几乎不可用。Stable Diffusion 3、Meta MovieGen 等全部依赖 CFG。

### 复习时不要混淆的两组概念

1. **Flow Matching vs Score Matching**
   前者直接回归向量场，后者回归 $\nabla \log p_t(x)$；对高斯路径二者可以互相换算。
2. **条件生成 vs Guidance**
   条件生成只是把 prompt 作为输入；guidance 则是在采样时主动放大 prompt 的影响，因此通常更“听话”。
