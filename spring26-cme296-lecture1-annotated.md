# [Spring 2026] CME 296 — Lecture 1: Diffusion Models (DDPM & DDIM)

> **来源：** `spring26-cme296-lecture1.pdf`（共 117 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [课程介绍与概述](#一课程介绍与概述)
2. [问题建模：从噪声生成图像](#二问题建模从噪声生成图像)
3. [前向过程（加噪）](#三前向过程加噪)
4. [变分推导：ELBO 与可处理的损失函数](#四变分推导elbo-与可处理的损失函数)
5. [训练算法](#五训练算法)
6. [推理算法](#六推理算法)
7. [DDPM 的局限性](#七ddpm-的局限性)
8. [加速采样：DDIM](#八加速采样ddim)

---

## 一、课程介绍与概述

### 1.1 教学团队

| 讲师 | 背景 | 工作经历 |
|------|------|---------|
| **Afshine Amidi** | Centrale Paris ('16), MIT ('17) | Uber, Google, Netflix |
| **Shervine Amidi** | Centrale Paris ('16), Stanford ('19) | Uber, Google, Netflix |

### 1.2 课程动机

生成式 AI 正在改变我们创造图像、视频和文本的方式。从 GAN（2014）到 ChatGPT（2026），生成模型的能力飞速提升。CME 296 的目标是：

1. **理解图像生成的核心范式**——我们如何从噪声中"雕刻"出逼真的图像
2. **学习图像生成模型的训练与使用方法**

### 1.3 先修知识

| 领域 | 所需知识 |
|------|---------|
| **线性代数** | 向量、矩阵、梯度、散度 |
| **概率论** | 贝叶斯法则、条件/边际概率、期望、协方差、高斯分布 |
| **微分方程** | 常微分方程（ODE）、随机微分方程（SDE）、数值求解器 |
| **机器学习基础** | 损失函数、训练、推理、神经网络 |

### 1.4 课程结构

本课程围绕图像生成的完整流程展开：

```
条件 (Condition) → 生成范式 (Generation paradigm) → 模型架构 (Model architecture)
→ 模型训练 (Model training) → 评估 (Evaluation)
```

- **L1–L3**：生成范式（Diffusion、Score Matching、Flow Matching）
- **L4**：模型架构
- **L5**：条件生成
- **L6**：模型训练
- **L7**：评估

> 📝 **注解：** 前三讲是整个课程的理论核心，分别介绍三种主流生成范式。本讲（L1）聚焦 Diffusion（扩散模型），从离散的 DDPM 出发，到加速采样的 DDIM。

### 1.5 符号约定

> ⚠️ 扩散模型领域存在大量不同的符号/约定。CME 296 将：
> 1. 使用最通用的符号
> 2. 在整个课程中保持一致
> 3. 在符号变化不直观时特别指出

> 📝 **注解：** 这是学习扩散模型时最常见的困惑来源。不同论文使用不同的符号体系：Ho et al. (DDPM) 使用 $\beta_t$ 表示噪声调度，Song et al. (Score SDE) 使用 $g(t)$ 表示扩散系数，Lipman et al. (Flow Matching) 使用 $\alpha_t, \beta_t$ 表示调度函数。本课程会尽量统一符号。

---

## 二、问题建模：从噪声生成图像

### 2.1 问题定义

**目标：** 从图像分布中生成新的图像。

假设我们有一组来自某个分布的观测样本 $z_1, \ldots, z_N$（训练数据集），我们希望生成看起来像是从同一分布中采样的新图像。

> 📝 **注解：** 数学上，我们有一个未知的数据分布 $p_{\text{data}}(z)$，训练集是它的有限样本。生成模型的目标是构造一个机制，使得我们能够从 $p_{\text{data}}$ 中采样——即 $z_{\text{new}} \sim p_{\text{data}}$。

### 2.2 无条件 vs 条件生成

本讲聚焦**无条件生成**（Unconditioned generation）——不给定任何提示，直接从数据分布中采样。条件生成（如给定文本提示生成图像）将在后续讲座中讨论。

### 2.3 为什么从噪声开始？

| 原因 | 解释 |
|------|------|
| **噪声容易采样** | 从标准高斯分布 $\mathcal{N}(0, I)$ 采样非常简单 |
| **引入随机性** | 不同的噪声样本会生成不同的图像，保证多样性 |
| **高斯分布性质好** | 线性变换封闭、解析表达式已知、数学处理方便 |

### 2.4 雕塑类比

> "The sculpture is already complete within the marble block, before I start my work. It is already there, I just have to chisel away the superfluous material." — Michelangelo

生成模型就像 Michelangelo 雕塑：图像已经"隐藏"在噪声中，我们只需要学会"凿去"多余的噪声。

> 📝 **注解：** 这个类比非常精妙。DDPM 的"去噪"过程就是逐步"凿去"噪声，让隐藏在噪声中的图像显现出来。但与雕塑不同的是，每次从不同的噪声出发，会"凿出"不同的图像。

### 2.5 扩散的直觉

**前向过程（加噪）：** 逐步向清晰图像添加噪声

$$z_0 \xrightarrow{+\text{noise}} z_1 \xrightarrow{+\text{noise}} z_2 \xrightarrow{+\text{noise}} \cdots \xrightarrow{+\text{noise}} z_T \approx \text{纯噪声}$$

- 我们**知道**前向过程的每一步如何加噪（由我们自己设计）

**反向过程（去噪）：** 逐步从噪声中恢复图像

$$z_T \xrightarrow{-\text{noise}} z_{T-1} \xrightarrow{-\text{noise}} \cdots \xrightarrow{-\text{noise}} z_0 \approx \text{清晰图像}$$

- 我们**想要学习**反向过程（这是模型需要学的）

> 📝 **注解：** 前向过程是人为设计的，完全已知；反向过程是未知的，需要通过训练来学习。这种"已知前向、学习反向"的不对称性是扩散模型的核心设计思路。

### 2.6 图像的向量表示

一张图片被表示为一个高维向量：

- **宽度** $W$ × **高度** $H$ × **3 个色彩通道**（RGB）
- 每个像素 = $(R, G, B)$，每个分量取值 0–255
- 整张图片 $= W \times H \times 3$ 维向量

> 📝 **注解：** 例如一张 256×256 的 RGB 图片对应 $d = 256 \times 256 \times 3 = 196{,}608$ 维向量。在扩散模型中，所有操作（加噪、去噪）都在这个高维空间中进行。

### 2.7 高维概率分布

在 1D 空间中，高斯分布由**均值**（标量）和**方差**（标量）参数化。

在高维空间中，高斯分布由**均值向量**和**协方差矩阵**参数化：

$$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

好消息：我们主要（甚至只）处理**各向同性高斯分布**（Isotropic Gaussian）：

$$\mathcal{N}(\boldsymbol{\mu}, \sigma^2 I)$$

其中协方差矩阵是对角矩阵 $\sigma^2 I$——所有方向上的方差相同，概率密度函数已知且"容易"处理。

> 📝 **注解：** 各向同性高斯分布的 PDF 为 $p(\mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{2\sigma^2}\right)$。关键性质：两个独立的各向同性高斯分布之和仍然是高斯分布（虽然不一定各向同性），这使得前向过程的数学推导变得可行。

---

## 三、前向过程（加噪）

### 3.1 噪声表示

噪声 $\epsilon \sim \mathcal{N}(0, I)$ 是**独立同分布**（i.i.d.）的标准高斯随机变量。

### 3.2 噪声调度

前向过程的核心是**噪声调度**（Noise Schedule）：从噪声图像到更噪声图像的变换规则。

$$\text{noisy image} + \text{noise} \xrightarrow{\text{noise schedule}} \text{noisier image}$$

噪声调度由我们自己定义，其分布已知。

### 3.3 前向过程的数学形式

**单步加噪：**

$$q(z_t | z_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\, z_{t-1},\; \beta_t I)$$

其中 $\beta_t$ 是第 $t$ 步的噪声方差（噪声调度）。

**等价采样（重参数化）：**

$$z_t = \sqrt{1 - \beta_t}\, z_{t-1} + \sqrt{\beta_t}\, \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

> 📝 **注解：** 这里 $\sqrt{1-\beta_t}$ 是缩放因子，$\sqrt{\beta_t}$ 是噪声的标准差。$\beta_t$ 控制每一步添加的噪声量——$\beta_t$ 越大，添加的噪声越多。典型的噪声调度是线性的 $\beta_t$ 从 0.0001 线性增长到 0.02，或者余弦调度。

### 3.4 任意时刻的直接采样

定义 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，则可以直接从 $z_0$ 采样 $z_t$：

$$q(z_t | z_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, z_0,\; (1 - \bar{\alpha}_t) I)$$

**等价采样：**

$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**推导：** 独立高斯分布之和的方差等于各自方差之和。

> 📝 **注解推导：** 逐步展开：
>
> $z_1 = \sqrt{\alpha_1} z_0 + \sqrt{\beta_1} \epsilon_0$
>
> $z_2 = \sqrt{\alpha_2} z_1 + \sqrt{\beta_2} \epsilon_1 = \sqrt{\alpha_1\alpha_2} z_0 + \sqrt{\alpha_2\beta_1}\epsilon_0 + \sqrt{\beta_2}\epsilon_1$
>
> 由于 $\epsilon_0, \epsilon_1$ 独立，噪声项的方差为 $\alpha_2\beta_1 + \beta_2 = \alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$。
>
> 一般地，$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，其中 $\epsilon$ 是等效噪声。
>
> 这个结果极为重要——它意味着我们可以在训练时**直接采样任意时刻** $t$ 的噪声图像，而不需要逐步迭代。这大大加速了训练过程。

---

## 四、变分推导：ELBO 与可处理的损失函数

### 4.1 学习目标

我们想要学习反向过程 $p_\theta(z_{t-1} | z_t)$，使得从噪声出发，逐步去噪后能得到逼真的图像。

**目标函数：** 找到参数 $\theta$，最大化训练图像在模型下的（对数）似然：

$$\max_\theta \sum_{i=1}^{N} \log p_\theta(z_0^{(i)})$$

> 📝 **注解：** 取对数是为了数值稳定性（避免下溢），同时利用对数的良好性质（如 $\log(ab) = \log a + \log b$）。最大化对数似然等价于最大化似然，因为 $\log$ 是单调递增函数。

### 4.2 推导策略

PDF 中给出了清晰的四步策略：

1. **推导 ELBO**（Evidence Lower BOund）——对数似然的下界
2. **展开 ELBO 各项**——分解为可处理的项
3. **证明 ELBO 可处理**——利用贝叶斯法则和高斯性质
4. **推导最终损失函数**——得到极其简单的噪声预测目标

### 4.3 Step 1：ELBO 推导

利用 Jensen 不等式，引入变分分布 $q(z_{1:T} | z_0)$（前向过程），得到：

$$\log p_\theta(z_0) \geq \underbrace{\mathbb{E}_{q(z_{1:T}|z_0)}\left[\log \frac{p_\theta(z_{0:T})}{q(z_{1:T}|z_0)}\right]}_{\text{ELBO}}$$

> 📝 **注解：** ELBO 是对数似然的下界——如果我们最大化 ELBO，就在"推高"对数似然。这是变分推断的核心思想：直接优化对数似然太难（因为需要计算 $p_\theta(z_0) = \int p_\theta(z_{0:T})\,\mathrm{d}z_{1:T}$，积分不可计算），转而优化它的下界。

### 4.4 概率论复习：联合分布、条件分布、边际化

在继续推导之前，PDF 复习了关键的概率概念：

**联合概率分布：** $p(x, y)$ 是同时观察到 $x$ 和 $y$ 的概率密度。

**条件概率：** $p(x | y)$ 是已知 $y$ 后 $x$ 的概率密度。

**边际化：** $p(x) = \int p(x, y)\,\mathrm{d}y = \int p(x|y)\,p(y)\,\mathrm{d}y$

> 📝 **注解：** 边际化公式就是全概率公式——将联合分布对某个变量积分，得到另一个变量的边际分布。在扩散模型中，我们经常需要将条件分布对隐变量积分来得到边际分布。

### 4.5 Step 2：展开 ELBO

展开 ELBO 后，得到若干项，其中包含**KL 散度**：

$$D_{\text{KL}}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)}\,\mathrm{d}x$$

KL 散度衡量两个分布之间的"距离"（非对称的），始终 $\geq 0$。

> 📝 **注解：** KL 散度在扩散模型中自然出现，因为 ELBO 的展开涉及比较前向分布 $q$ 和反向分布 $p_\theta$。KL 散度越小，说明两个分布越接近。

### 4.6 Step 3：证明 ELBO 可处理

展开后的 ELBO 包含两类项需要证明可处理：

**(a) 前向过程的后验 $q(z_{t-1} | z_t, z_0)$：**

利用贝叶斯法则 + 马尔可夫性质，可以证明：

$$q(z_{t-1} | z_t, z_0) = \mathcal{N}\left(\tilde{\boldsymbol{\mu}}_t(z_t, z_0),\; \tilde{\beta}_t I\right)$$

其中：

$$\tilde{\boldsymbol{\mu}}_t(z_t, z_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t} z_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} z_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

> 📝 **注解推导：** 由贝叶斯法则：
>
> $$q(z_{t-1}|z_t, z_0) = \frac{q(z_t|z_{t-1}, z_0)\,q(z_{t-1}|z_0)}{q(z_t|z_0)}$$
>
> 由马尔可夫性质：$q(z_t|z_{t-1}, z_0) = q(z_t|z_{t-1})$
>
> 三个高斯分布的比值仍然是高斯分布，通过配方法可以求出均值和方差。

**(b) 反向过程的参数化 $p_\theta(z_{t-1} | z_t)$：**

我们假设反向过程也是高斯的：

$$p_\theta(z_{t-1} | z_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(z_t, t),\; \sigma_t^2 I)$$

> 📝 **注解：** 为什么可以假设反向过程是高斯的？直觉上，如果前向过程的每一步只添加少量噪声（$\beta_t$ 很小），那么反向过程的每一步也只是做微小的调整，近似高斯是合理的。这是 DDPM 论文中的关键假设。

### 4.7 Step 4：推导最终损失函数

计算两个高斯分布之间的 KL 散度，经过化简后，损失函数变为：

$$\mathcal{L}_t = \mathbb{E}_{z_0, \epsilon}\left[\left\| \epsilon - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\,z_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\; t) \right\|^2\right]$$

> 🎯 **极其简单的结果！** 损失函数就是：**预测噪声** $\epsilon$ vs **实际噪声** $\epsilon_\theta$ 的均方误差。

> 📝 **注解推导：** 两个高斯分布 $q = \mathcal{N}(\mu_1, \sigma^2 I)$ 和 $p = \mathcal{N}(\mu_2, \sigma^2 I)$ 之间的 KL 散度为：
>
> $$D_{\text{KL}}(q \| p) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$$
>
> 将 $q$ 的均值 $\tilde{\mu}_t(z_t, z_0)$ 和 $p_\theta$ 的均值 $\mu_\theta(z_t, t)$ 代入，并将 $z_t$ 用 $z_0$ 和 $\epsilon$ 表示，经过代数化简，最终得到噪声预测损失。
>
> 关键步骤：将 $\mu_\theta$ 参数化为预测噪声的形式 $\epsilon_\theta$，即：
>
> $$\mu_\theta(z_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(z_t, t)\right)$$
>
> 这样 KL 散度就简化为 $\|\epsilon - \epsilon_\theta\|^2$。

### 4.8 推导总结

```
策略：
1. 推导 ELBO（对数似然下界）         ← Jensen 不等式
2. 展开 ELBO 各项                     ← 对数性质 + 重组
3. 证明 ELBO 可处理                   ← 贝叶斯法则 + 高斯性质
4. 推导最终损失函数                    ← 两个高斯的 KL 散度

结果：极其简单的噪声预测！
```

---

## 五、训练算法

### 5.1 训练步骤

```
重复以下步骤直到收敛：

1. 采样：
   - 干净图像 z_0 ~ p_data（从数据集中取一张图片）
   - 时间步 t ~ Uniform{1, ..., T}
   - 噪声 ε ~ N(0, I)

2. 构造噪声图像：
   z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε

3. 计算损失并反向传播：
   L = ‖ε - ε_θ(z_t, t)‖²
   通过 θ 进行梯度下降更新
```

> 📝 **注解：** 训练过程非常简洁——每次迭代只需要：
> 1. 从数据集取一张图
> 2. 随机选一个时间步
> 3. 生成噪声并构造噪声图像（利用 3.4 节的直接采样公式）
> 4. 让神经网络预测噪声，计算与真实噪声的 MSE 损失
> 5. 反向传播更新参数
>
> 注意：不需要逐步迭代前向过程！直接采样任意时刻 $t$ 的噪声图像是 DDPM 训练效率的关键。

---

## 六、推理算法

### 6.1 推理步骤

```
1. 采样纯噪声：z_T ~ N(0, I)

2. 从 t = T 到 t = 1 迭代去噪：
   z_{t-1} = μ_θ(z_t, t) + σ_t · ε'
   
   其中：
   - μ_θ(z_t, t) = (1/√α_t)(z_t - (β_t/√(1-ᾱ_t)) · ε_θ(z_t, t))
   - ε' ~ N(0, I)（随机噪声，t > 1 时添加；t = 1 时不添加）
   - ε_θ(z_t, t) 是神经网络预测的噪声

3. 返回 z_0（生成的图像）
```

> 📝 **注解：** 推理过程是从纯噪声 $z_T$ 开始，逐步去噪到 $z_0$。每一步：
> 1. 神经网络预测当前噪声图像中的噪声 $\epsilon_\theta(z_t, t)$
> 2. 利用预测的噪声计算去噪后的均值 $\mu_\theta$
> 3. 添加少量随机噪声 $\sigma_t \epsilon'$（除了最后一步），保证生成多样性
>
> 添加随机噪声的原因：反向过程本身是随机的（SDE），保留随机性可以避免模式坍塌，提高生成多样性。

---

## 七、DDPM 的局限性

### 7.1 里程碑论文

> **Denoising Diffusion Probabilistic Models**, Ho et al., 2020.

DDPM 是扩散模型的开山之作，让扩散模型重新成为生成模型的主流。

### 7.2 核心问题：推理太慢

DDPM 的生成过程需要 **$T = 1000$ 步**迭代，每步都需要运行一次神经网络前向传播。

- 比 VAE/GAN 慢 **几个数量级**
- 可能每张图片需要花费 **数分钟** 生成

> 📝 **注解：** 这是 DDPM 最严重的实际缺陷。VAE 和 GAN 只需要一次前向传播就能生成图片，而 DDPM 需要 1000 次。虽然生成质量极高，但推理速度严重限制了实际应用。

---

## 八、加速采样：DDIM

### 8.1 缓解尝试

**尝试 1：** 通过归纳法推导跳步公式？

问题：仍然需要多次函数评估，没有实质进步。

**尝试 2：** 直接跳步？

问题：大步跳跃 + 随机性 = 质量很差。

### 8.2 关键洞察

DDPM 的损失函数**只依赖于边际分布** $q(z_t | z_0)$，不依赖于前向过程的具体马尔可夫结构。

这意味着：
- 边际分布与 DDPM 相同
- 生成过程可以变为**确定性的**

> 📝 **注解：** 这是 DDIM 的理论基础。DDPM 的训练目标 $\|\epsilon - \epsilon_\theta(z_t, t)\|^2$ 只涉及 $z_t$ 和 $z_0$（边际量），不涉及中间步骤 $z_1, \ldots, z_{t-1}$。因此，我们可以设计一个不同的（非马尔可夫的）前向过程，只要边际分布相同，训练好的模型就可以直接使用。

### 8.3 问题重构

关键性质：生成过程可以变为确定性的。

通过归纳法可以证明：在保持边际分布不变的前提下，可以去掉中间步骤的随机性。

### 8.4 DDIM 的含义

**DDIM = Denoising Diffusion Implicit Models**

DDIM 的更新规则：

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\underbrace{\hat{z}_0}_{\text{"我们对干净图像的预测"}} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \underbrace{\epsilon_\theta(z_t, t)}_{\text{预测的噪声}}$$

其中 $\hat{z}_0$ 是从时刻 $t$ 预测的干净图像：

$$\hat{z}_0 = \frac{z_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}}$$

> 📝 **注解：** DDIM 的更新规则非常直观：
> 1. 从当前噪声图像 $z_t$ 和预测噪声 $\epsilon_\theta$，估计出干净图像 $\hat{z}_0$
> 2. 然后用 $\hat{z}_0$ 和预测噪声重新构造 $z_{t-1}$
> 3. **没有随机噪声项！** 整个过程是确定性的
>
> 与 DDPM 的对比：DDPM 每步添加随机噪声 $\sigma_t \epsilon'$，DDIM 不添加。

### 8.5 DDIM 的加速效果

**更新规则：** 可以跳步！

$$z_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{z}_0 + \sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\,\epsilon_\theta(z_{\tau_i}, \tau_i)$$

其中 $\tau_1 < \tau_2 < \cdots < \tau_S$ 是从 $\{1, \ldots, T\}$ 中选取的子序列，$S \ll T$。

**加速效果：**

| 加速倍数 | 1x | 10x | 20x | 50x | 100x |
|---------|-----|------|------|------|-------|
| FID 影响 | 基线 | +3% | +16% | +70% | +330% |

> 📝 **注解：** FID（Fréchet Inception Distance）越低越好。+3% 表示 FID 增加了 3%（质量略微下降）。可以看到：
> - 10x 加速（100 步→10 步）：质量几乎不变
> - 20x 加速（1000 步→50 步）：质量下降 16%，仍然可接受
> - 50x 以上：质量显著下降
>
> 实践中，10x-20x 加速是质量与速度的最佳平衡点。

### 8.6 DDIM 结论

**DDIM 的三步法：**

1. 选择与 DDPM 优化兼容的"方便"建模假设
2. 去掉中间步骤的随机性 → DDIM
3. 跳步！

> 📝 **注解：** DDIM 的核心贡献是发现了一个非马尔可夫的前向过程，其边际分布与 DDPM 相同，因此可以复用 DDPM 训练好的模型权重，无需重新训练。这使得 DDIM 成为一种"免费"的加速方法——只需要改变推理方式，不需要改变训练方式。

### 8.7 与后续讲座的联系

DDIM 的确定性采样本质上是将 SDE 采样改为了 ODE 采样。这个 ODE 就是**概率流 ODE（PF-ODE）**，将在 Lecture 2 中详细讨论。

> 📝 **注解：** DDIM 可以看作 PF-ODE 的离散化近似。Lecture 2 将从 SDE 的角度统一 DDPM 和 NCSN，并推导出 PF-ODE 和 DPM-Solver 等更高效的采样方法。

---

## 附录：关键公式速查表

| 符号 | 定义 | 含义 |
|------|------|------|
| $\beta_t$ | 噪声调度 | 第 $t$ 步添加的噪声方差 |
| $\alpha_t$ | $1 - \beta_t$ | 信号保留比例 |
| $\bar{\alpha}_t$ | $\prod_{s=1}^t \alpha_s$ | 累积信号保留比例 |
| $z_t$ | $\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | 时刻 $t$ 的噪声图像 |
| $\epsilon_\theta(z_t, t)$ | 神经网络 | 预测时刻 $t$ 的噪声 |
| $\mu_\theta(z_t, t)$ | $\frac{1}{\sqrt{\alpha_t}}(z_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta)$ | 反向过程均值 |
| $\tilde{\mu}_t(z_t, z_0)$ | 前向后验均值 | 前向过程的后验分布均值 |
| $\hat{z}_0$ | $\frac{z_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$ | 从 $z_t$ 预测的干净图像 |
