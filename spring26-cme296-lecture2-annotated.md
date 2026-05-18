# [Spring 2026] CME 296 — Lecture 2: Score Matching, SDE & Probability Flow ODE

> **来源：** `spring26-cme296-lecture2.pdf`（共 121 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [上讲回顾](#一上讲回顾)
2. [Score 函数的直觉](#二score-函数的直觉)
3. [Score 估计：Score Matching](#三score-估计score-matching)
4. [微分形式化：从离散到连续的 SDE](#四微分形式化从离散到连续的-sde)
5. [训练](#五训练)
6. [推理：反向 SDE](#六推理反向-sde)
7. [概率流 ODE（PF-ODE）](#七概率流-odepf-ode)
8. [DPM-Solver：专为 PF-ODE 设计的求解器](#八dpm-solver专为-pf-ode-设计的求解器)

---

## 一、上讲回顾

### 1.1 DDPM 的核心思路

Lecture 1 建立了扩散模型的基本框架：

**目标：** 从图像分布中生成新的图像。

**三步策略：**

1. **加噪（前向过程）**：逐步向清晰图像添加噪声
2. **学习去噪（反向过程）**：训练神经网络预测噪声
3. **推理**：从纯噪声出发，逐步去噪生成图像

### 1.2 变分推导的四步法

```
1. 推导 ELBO（对数似然下界）         ← Jensen 不等式
2. 展开 ELBO 各项                     ← 对数性质 + 重组
3. 证明 ELBO 可处理                   ← 贝叶斯法则 + 高斯性质
4. 推导最终损失函数                    ← 两个高斯的 KL 散度
```

**结果：** 极其简单的噪声预测损失 $\mathcal{L} = \|\epsilon - \epsilon_\theta(z_t, t)\|^2$

> 📝 **注解：** DDPM 的训练目标是预测噪声 $\epsilon$，推理时用预测的噪声来计算去噪后的图像。本讲将介绍另一种视角——**Score Matching**，它预测的不是噪声，而是**概率密度的梯度**（score 函数）。这两种视角在 SDE 框架下被统一。

---

## 二、Score 函数的直觉

### 2.1 本讲主题

Lecture 2 的主题是 **Score Matching**（分数匹配），这是与 DDPM 并行的另一种生成范式。

> 📝 **注解：** DDPM 从"噪声预测"角度出发，Score Matching 从"梯度估计"角度出发。它们看似不同，但在 SDE 框架下被证明是等价的。这种统一视角是本讲的核心。

### 2.2 梯度的直觉

**梯度** $\nabla_x f(x)$ 指向函数增长最快的方向。

对于概率密度函数 $p(x)$，梯度 $\nabla_x p(x)$ 指向概率密度增加最快的方向——即"更可能"出现数据的方向。

### 2.3 概率梯度的问题

直接使用 $\nabla_x p(x)$ 有两个严重问题：

**问题 1：不可计算**

概率密度的归一化常数 $Z = \int p(x)\,\mathrm{d}x$ 不可计算：

$$p(x) = \frac{\tilde{p}(x)}{Z}$$

因此 $\nabla_x p(x) = \frac{\nabla_x \tilde{p}(x)}{Z}$，但 $Z$ 未知。

**问题 2：低密度区域数值不稳定**

在 $p(x)$ 很小的区域，$\nabla_x p(x)$ 的估计非常不可靠。

> 📝 **注解：** 这两个问题是统计推断中的经典难题。归一化常数 $Z$（配分函数）在高维空间中几乎不可能精确计算。低密度区域的问题更微妙——数据稀疏的地方，梯度估计的方差极大，导致 Langevin 采样在这些区域"迷路"。

### 2.4 解决方案：对数概率的梯度

**提议：** 考虑 $\nabla_x \log p(x)$ 代替 $\nabla_x p(x)$

$$\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

**三大优势：**

1. **不再不可计算！** 归一化常数 $Z$ 在对数梯度中消失：

$$\nabla_x \log p(x) = \nabla_x \log \frac{\tilde{p}(x)}{Z} = \nabla_x \log \tilde{p}(x) - \underbrace{\nabla_x \log Z}_{=0} = \nabla_x \log \tilde{p}(x)$$

2. **方向与 $\nabla_x p(x)$ 相同**——只是幅度不同，方向一致

3. **数值更稳定**——对数变换压缩了动态范围

> 📝 **注解：** $\nabla_x \log Z = 0$ 因为 $Z$ 是常数（与 $x$ 无关）。这是 score matching 的核心洞察——通过对数变换，我们绕过了归一化常数的计算难题。这个技巧在统计物理学中也有广泛应用（如配分函数的对数导数）。

### 2.5 Score 函数的定义

$$s(x) := \nabla_x \log p(x)$$

**Score 函数** = 对数概率密度的梯度。

它告诉我们：在位置 $x$，向哪个方向走能让概率密度增加最快。

> 📝 **注解：** Score 函数的直觉：想象你站在一个概率密度地形上，score 函数指向"上坡"方向——即数据更密集的方向。如果我们沿着 score 函数走，就能从低密度区域（噪声）走到高密度区域（数据）。

### 2.6 Score 函数的直觉

Score 函数在数据密集的区域指向数据聚类的中心，在数据稀疏的区域提供"导航"方向。如果知道了 score 函数，就可以用 **Langevin 采样**从分布中采样。

> 📝 **注解：** Langevin 采样的更新规则为 $x_{t+1} = x_t + \eta\, s(x_t) + \sqrt{2\eta}\,\epsilon$，其中 $\eta$ 是步长，$\epsilon \sim \mathcal{N}(0, I)$。这本质上就是沿着 score 方向走一小步，再加上一点随机噪声。在合适的条件下，Langevin 采样会收敛到目标分布。

---

## 三、Score 估计：Score Matching

### 3.1 目标

**目标：** 用神经网络 $s_\theta(x)$ 估计 score 函数 $s(x) = \nabla_x \log p(x)$。

问题：我们**无法直接计算** $s(x)$，因为 $p(x)$ 未知。

### 3.2 估计 Score 的尝试

**方法 1：Implicit Score Matching (ISM)**

> *Estimation of Non-Normalized Statistical Models by Score Matching*, Hyvärinen, 2005.

通过分部积分，将需要计算 $s(x)$ 的问题转化为只需计算 $\nabla_x s_\theta(x)$ 的问题。

**方法 2：Sliced Score Matching (SSM)**

> *Sliced Score Matching: A Scalable Approach to Density and Score Estimation*, Song et al., 2019.

将高维 score 投影到随机向量上，降低计算复杂度。

> 📝 **注解：** ISM 通过分部积分避免了计算 $s(x)$，但需要计算 $\nabla_x \cdot s_\theta(x)$（score 的散度），这在高维空间中计算代价很大。SSM 通过随机投影降低了维度，但引入了额外的方差。这两种方法都有局限性。

### 3.3 高斯分布的 Score（1D 例子）

对于 $p(x) = \mathcal{N}(\mu, \sigma^2)$：

- 概率密度函数：$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
- Score：$s(x) = \nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}$

> 📝 **注解：** 高斯分布的 score 非常简单——它是一个线性函数，指向均值 $\mu$ 的方向，幅度与到均值的距离成正比，与方差 $\sigma^2$ 成反比。这意味着：1) 离均值越远，"拉力"越强；2) 方差越大，"拉力"越弱。这个结果在后续的 Denoising Score Matching 中至关重要。

### 3.4 新思路：给数据加噪

**想法：** 给数据加噪声，利用加噪后分布的 score 的解析表达式。

定义加噪分布：

$$p_\sigma(x) = \int p(x | z)\, p_{\text{data}}(z)\,\mathrm{d}z$$

其中 $p(x|z) = \mathcal{N}(z, \sigma^2 I)$。

由于 $p(x|z)$ 是高斯的，加噪分布的 score 可以解析计算：

$$\nabla_x \log p_\sigma(x) = \frac{\mathbb{E}_{z \sim p_{\text{data}}(\cdot|x)}[z - x]}{\sigma^2}$$

> 📝 **注解：** 这是利用高斯分布 score 的解析性质。加噪后的分布是数据分布与高斯的卷积，其 score 可以通过条件期望表示。关键洞察：加噪使得分布更平滑，score 更容易估计。

### 3.5 Denoising Score Matching (DSM)

> *A Connection Between Score Matching and Denoising Autoencoders*, Vincent, 2010.

通过展开平方范数并利用边际分布的定义，可以证明：

$$\mathbb{E}_{p_\sigma(x)}\left[\|s_\theta(x) - \nabla_x \log p_\sigma(x)\|^2\right] = \mathbb{E}_{z, \epsilon}\left[\left\|s_\theta(z + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\|^2\right] + C$$

其中 $C$ 是与 $\theta$ 无关的常数。

**DSM 损失函数（可处理！）：**

$$\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I)}\left[\left\|s_\theta(z + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\|^2\right]$$

> 📝 **注解：** DSM 的损失函数完全可计算——不需要知道 $p_\sigma(x)$ 或其 score。我们只需要：
> 1. 从数据集采样 $z$
> 2. 从标准正态采样 $\epsilon$
> 3. 构造加噪样本 $x = z + \sigma\epsilon$
> 4. 计算神经网络预测 $s_\theta(x)$ 与目标 $-\epsilon/\sigma$ 的 MSE
>
> 目标 $-\epsilon/\sigma$ 就是加噪分布的 score：$\nabla_x \log p_\sigma(x) = -\epsilon/\sigma$（当 $x = z + \sigma\epsilon$ 时）。

### 3.6 Vanilla DSM 的局限

**问题：** $p_\sigma(x) \neq p_{\text{data}}(x)$

- **小 $\sigma$**：$p_\sigma \approx p_{\text{data}}$，但在低密度区域 score 估计很差
- **大 $\sigma$**：低密度区域 score 估计好，但 $p_\sigma$ 远离 $p_{\text{data}}$

> 📝 **注解：** 这是 DSM 的根本矛盾——单一噪声水平无法同时满足两个要求：
> 1. 加噪后的分布要接近真实数据分布（需要小 $\sigma$）
> 2. 低密度区域的 score 估计要可靠（需要大 $\sigma$）
>
> 解决方案：使用**多个噪声水平**！

### 3.7 NCSN：多噪声水平的 Score Matching

> *Generative Modeling by Estimating Gradients of the Data Distribution*, Song et al., 2019.

**NCSN = Noise Conditional Score Network**

**想法：** 在不同噪声水平上估计 score。

使用一系列递减的噪声水平 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$，训练一个条件网络 $s_\theta(x, \sigma_i)$。

**损失函数：**

$$\mathcal{L}_{\text{NCSN}}(\theta) = \frac{1}{L}\sum_{i=1}^{L} \mathbb{E}_{z, \epsilon}\left[\left\|s_\theta(z + \sigma_i\epsilon, \sigma_i) + \frac{\epsilon}{\sigma_i}\right\|^2\right]$$

> 📝 **注解：** NCSN 的核心创新是使用多个噪声水平。大噪声水平覆盖低密度区域，小噪声水平保持分布精度。这就像使用不同分辨率的"放大镜"来观察数据分布——粗略的看整体结构，精细的看局部细节。

### 3.8 退火 Langevin 动力学（ALD）采样

训练好 NCSN 后，使用 **Annealed Langevin Dynamics (ALD)** 采样：

```
1. 采样初始噪声：x ~ N(0, σ_1² I)（最大噪声水平）

2. 对每个噪声水平 σ_i（从大到小）：
   执行 T_i 步 Langevin 更新：
   
   x ← x + η_i · s_θ(x, σ_i) + √(2η_i) · ε
   
   其中 η_i 是步长，ε ~ N(0, I)

3. 返回 x（最终生成的图像）
```

> 📝 **注解：** ALD 的直觉：先在大噪声水平下快速移动到高密度区域（粗定位），然后逐步减小噪声水平进行精细调整（精定位）。这类似于模拟退火——先在高温下自由探索，然后逐步降温锁定最优解。

### 3.9 DDPM 与 NCSN 的平行对比

| | DDPM | NCSN |
|---|---|---|
| **预测目标** | 从噪声样本中去除的噪声 | 从噪声到干净的 score |
| **训练损失** | $\|\epsilon - \epsilon_\theta\|^2$ | $\|s_\theta + \epsilon/\sigma\|^2$ |
| **采样方式** | 逐步去噪 | 退火 Langevin 动力学 |

> 📝 **注解：** DDPM 预测噪声 $\epsilon$，NCSN 预测 score $s = -\epsilon/\sigma^2$。两者之间存在简单的线性关系！这暗示它们可能是同一事物的不同参数化。Lecture 2 的后半部分将用 SDE 框架统一这两种方法。

---

## 四、微分形式化：从离散到连续的 SDE

### 4.1 动机

DDPM 使用离散时间步 $t = 1, 2, \ldots, T$，NCSN 使用任意噪声水平 $\sigma_1 > \cdots > \sigma_L$。

**关键问题：** 如果我们考虑**连续演化**而不是离散步骤呢？

> 📝 **注解：** 连续时间形式化有两个好处：1) 统一 DDPM 和 NCSN——它们只是同一 SDE 的不同离散化；2) 利用 SDE 理论的丰富工具（如 Anderson 反向定理、Fokker-Planck 方程）来推导新结果。

### 4.2 Wiener 过程（布朗运动）

> *Score-Based Generative Modeling through Stochastic Differential Equations*, Song et al., 2020.

**Wiener 过程** $W_t$ 是连续时间中添加高斯噪声增量的等价物。

**特殊性质：**
- **独立增量**：$W_{t+s} - W_t$ 与过去无关
- **正态增量**：$W_{t+s} - W_t \sim \mathcal{N}(0, sI)$
- **连续路径**：$W_t$ 的轨迹几乎必然连续

> 📝 **注解：** Wiener 过程是布朗运动的数学模型。它的核心特征是增量的方差与时间长度成正比（不是时间长度的平方）。这使得 $\mathrm{d}W_t$ 的"大小"为 $\sqrt{\mathrm{d}t}$ 量级——这是 SDE 与 ODE 在数学处理上最大的不同。

### 4.3 DDPM 的连续形式化

> *Score-Based Generative Modeling through Stochastic Differential Equations*, Song et al., 2020.

**离散 → 连续：**

DDPM 的离散前向过程 $z_t = \sqrt{1-\beta_t}z_{t-1} + \sqrt{\beta_t}\epsilon$ 在连续极限下变为：

$$\mathrm{d}x = -\frac{1}{2}\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t$$

> 📝 **注解推导：** 令 $\Delta t = 1/T$，$\beta_t = \beta(t)\Delta t$。则：
>
> $z_t = \sqrt{1-\beta_t}z_{t-1} + \sqrt{\beta_t}\epsilon \approx (1 - \frac{1}{2}\beta_t)z_{t-1} + \sqrt{\beta_t}\epsilon$
>
> $z_t - z_{t-1} = -\frac{1}{2}\beta(t)\Delta t \cdot z_{t-1} + \sqrt{\beta(t)\Delta t}\cdot\epsilon$
>
> 当 $\Delta t \to 0$ 时，$\sqrt{\Delta t}\cdot\epsilon \to \mathrm{d}W_t$，得到上述 SDE。
>
> 这就是 **Variance Preserving (VP) SDE**——因为前向过程中 $x$ 的方差保持有界。

### 4.4 NCSN 的连续形式化

NCSN 的离散加噪过程 $x = z + \sigma\epsilon$ 在连续极限下变为：

$$\mathrm{d}x = \sqrt{\frac{\mathrm{d}[\sigma^2(t)]}{\mathrm{d}t}}\,\mathrm{d}W_t$$

> 📝 **注解：** 这就是 **Variance Exploding (VE) SDE**——没有漂移项（纯噪声添加），方差随时间无限增长。与 VP-SDE 的区别：VP-SDE 有漂移项将信号缩放，VE-SDE 只添加噪声不缩放信号。

### 4.5 SDE 的一般形式

$$\mathrm{d}x = \underbrace{f(x, t)}_{\text{漂移（drift）}}\,\mathrm{d}t + \underbrace{g(t)}_{\text{扩散（diffusion）}}\,\mathrm{d}W_t$$

| 项 | 性质 | 作用 |
|---|---|---|
| $f(x, t)\,\mathrm{d}t$ | **确定性** | 系统的"推力" |
| $g(t)\,\mathrm{d}W_t$ | **随机性** | 系统的"抖动" |

> 📝 **注解：** 这是 SDE 的标准 Itô 形式。漂移项 $f(x,t)$ 决定了系统的确定性演化方向，扩散项 $g(t)\mathrm{d}W_t$ 引入随机性。在扩散模型中，前向 SDE 的漂移项将数据推向噪声，扩散项控制噪声的强度。

### 4.6 SDE 变体

| 变体 | SDE | 对应方法 |
|------|-----|---------|
| **Variance Preserving (VP)** | $\mathrm{d}x = -\frac{1}{2}\beta(t)x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t$ | DDPM |
| **Variance Exploding (VE)** | $\mathrm{d}x = \sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}}\,\mathrm{d}W_t$ | NCSN |

> 📝 **注解：** VP-SDE 之所以叫"方差保持"，是因为在漂移项 $-\frac{1}{2}\beta(t)x$ 的缩放下，$x$ 的方差不会无限增长，而是趋于稳定值。VE-SDE 之所以叫"方差爆炸"，是因为没有漂移项缩放，纯噪声添加导致方差无限增长。两种 SDE 在实践中各有优劣。

### 4.7 方法的时间线与统一

```
2006        2010        2015        2019        2020           2020
Score       Denoising   Diffusion   NCSN        DDPM           Score SDE
Matching    Score       (Sohl-                  (Ho et al.)    (Song et al.)
(Hyvärinen) Matching    Dickstein)                            
             (Vincent)   (Sohl-Dickstein)                      
                                                                
Principle ←────── Intuition ──────────────────────→ SDE 统一
```

> 📝 **注解：** Score SDE (2020) 是统一性工作——它将 DDPM 和 NCSN 统一到 SDE 框架下，揭示了它们只是同一 SDE 的不同离散化方式。这个统一视角不仅理论上优美，还催生了新的方法（如 PF-ODE、DPM-Solver）。

---

## 五、训练

### 5.1 训练推导

**目标：** 估计 score 函数 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$

> 📝 **注解：** 与 DDPM 训练预测噪声 $\epsilon_\theta$ 不同，SDE 框架下训练预测 score $s_\theta$。但两者之间存在简单的转换关系（见 3.9 节的对比）。

### 5.2 VP-SDE (DDPM 风格) 的训练目标

对于 VP-SDE，score 与噪声预测之间存在关系：

$$s_\theta(x, t) = -\frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar{\alpha}_t}}$$

因此 VP-SDE 的训练等价于 DDPM 的噪声预测训练。

### 5.3 VE-SDE (NCSN 风格) 的训练目标

对于 VE-SDE，训练目标为 Denoising Score Matching：

$$\mathcal{L}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\lambda(t)\left\|s_\theta(z + \sigma_t\epsilon, t) + \frac{\epsilon}{\sigma_t}\right\|^2\right]$$

其中 $\lambda(t)$ 是权重函数。

> 📝 **注解：** 权重函数 $\lambda(t)$ 的选择很重要。Karras et al. (2022) 在 *"Elucidating the Design Space of Diffusion-Based Generative Models"* 中对此有深入讨论。不同的权重选择会影响不同噪声水平上的训练质量。

### 5.4 训练算法

```
重复以下步骤直到收敛：

1. 采样：
   - 干净图像 z ~ p_data
   - 时间步 t ~ Uniform
   - 噪声 ε ~ N(0, I)

2. 构造加噪样本：
   - VP: x_t = √ᾱ_t · z + √(1-ᾱ_t) · ε
   - VE: x_t = z + σ_t · ε

3. 计算损失并反向传播：
   - VP: L = ‖ε_θ(x_t, t) - ε‖²
   - VE: L = ‖s_θ(x_t, t) + ε/σ_t‖²
```

> 📝 **注解：** VP 和 VE 的训练算法结构完全相同——都是采样数据、加噪、预测目标、计算 MSE。唯一区别在于参数化方式（预测噪声 vs 预测 score）和加噪方式（缩放+加噪 vs 纯加噪）。

---

## 六、推理：反向 SDE

### 6.1 反向 SDE 公式

> *Reverse-time diffusion equation models*, Anderson, 1982.

**前向 SDE：** $\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}W_t$（将数据扰动为噪声）

**反向 SDE：** 将噪声变换回数据

$$\mathrm{d}x = \left[f(x,t) - g^2(t)\,\nabla_x \log p_t(x)\right]\mathrm{d}t + g(t)\,\mathrm{d}\bar{W}_t$$

其中 $\bar{W}_t$ 是反向时间的 Wiener 过程。

> 📝 **注解：** Anderson (1982) 的定理给出了前向 SDE 的反向时间方程。反向 SDE 的漂移项包含三个部分：
> 1. $f(x,t)$：前向漂移（方向不变）
> 2. $-g^2(t)\nabla_x\log p_t(x)$：**score 修正项**（关键！）
> 3. $g(t)\mathrm{d}\bar{W}_t$：反向时间的随机噪声

### 6.2 反向 SDE 的直觉

反向 SDE 的漂移项中，score 函数 $\nabla_x \log p_t(x)$ 起到了**修正**的作用：

- 前向过程中，扩散项将粒子推向低密度区域
- 反向过程中，需要额外向高密度区域推以补偿扩散
- Score 函数恰好提供了这个"补偿方向"——它指向概率密度增加最快的方向

> 📝 **注解：** 可以这样理解：前向 SDE 的扩散效应让粒子"散开"，反向 SDE 需要让粒子"聚拢"。Score 函数就是"聚拢力"——它告诉粒子应该向哪个方向移动才能到达高密度区域。没有 score 修正项，反向 SDE 只会重复前向过程（继续散开），而不是逆转它。

### 6.3 推理算法

```
1. 采样纯噪声：x_T ~ N(0, I)（或对应前向 SDE 的终态分布）

2. 使用 Euler-Maruyama 方法求解反向 SDE：
   x_{t-Δt} = x_t + [f(x_t, t) - g²(t) · s_θ(x_t, t)] · Δt + g(t) · √Δt · ε
   
   其中 ε ~ N(0, I)，s_θ 是训练好的 score 网络

3. 返回 x_0（生成的图像）
```

> 📝 **注解：** 反向 SDE 的数值求解使用 Euler-Maruyama 方法，与 Lecture 1 中 DDPM 的逐步去噪本质相同。区别在于：SDE 框架给出了连续时间的理论，而 DDPM 是离散的特例。

---

## 七、概率流 ODE（PF-ODE）

### 7.1 SDE 的局限

随机项意味着：

- **求解器更慢**：需要 1000-2000 步，因为我们不知道哪个区域"容易"vs"困难"
- **更多误差来源**：既有有限步长的离散化误差，又有注入的随机噪声

### 7.2 假设：将 SDE 写成 ODE

如果没有随机项：

- **求解器更快**：可以利用数学性质决定哪里快走、哪里慢走
- **误差来源更集中**：只有有限步长贡献误差

> 📝 **注解：** ODE 的确定性使得我们可以使用自适应步长求解器（如 Runge-Kutta）——在变化剧烈的区域用小步长，在平缓的区域用大步长。SDE 的随机性使得这种自适应策略不可行。

### 7.3 PF-ODE 的推导

> *Score-Based Generative Modeling through Stochastic Differential Equations*, Song et al., 2020.

**推导路径：**

```
前向 SDE → Fokker-Planck 方程 → 连续性方程 → PF-ODE
```

**Step 1：** 前向 SDE 对应的 **Fokker-Planck 方程**描述了概率密度的时间演化：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (f \cdot p_t) + \frac{1}{2}g^2(t)\,\nabla_x^2 p_t$$

**Step 2：** 通过代数运算，将 Fokker-Planck 方程改写为**连续性方程**：

$$\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t \cdot \tilde{f}) = 0$$

其中 $\tilde{f}$ 是修正后的漂移。

**Step 3：** 识别出概率流的速度，得到 **PF-ODE**：

$$\mathrm{d}x = \left[f(x,t) - \frac{1}{2}g^2(t)\,\nabla_x \log p_t(x)\right]\mathrm{d}t$$

> 📝 **注解推导：** Fokker-Planck 方程中，$\nabla_x^2 p_t = \nabla_x \cdot (\nabla_x p_t) = \nabla_x \cdot (p_t \nabla_x \log p_t)$。将此项移到漂移项中：
>
> $$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (f \cdot p_t) + \frac{1}{2}g^2 \nabla_x \cdot (p_t \nabla_x \log p_t)$$
>
> $$= -\nabla_x \cdot \left[p_t\left(f - \frac{1}{2}g^2 \nabla_x \log p_t\right)\right]$$
>
> 这就是连续性方程，对应 ODE $\mathrm{d}x = \left[f - \frac{1}{2}g^2 \nabla_x \log p_t\right]\mathrm{d}t$。

### 7.4 PF-ODE 的含义

**PF-ODE = Probability Flow - Ordinary Differential Equation**

PF-ODE 满足：**与反向 SDE 具有相同的边际密度** $p_t(x)$

> ⚠️ **相同的边际密度 ≠ 相同的轨迹**

> 📝 **注解：** 这是 PF-ODE 最重要的性质——虽然 PF-ODE 的轨迹是确定性的，反向 SDE 的轨迹是随机的，但它们在任意时刻 $t$ 的概率密度分布完全相同。这意味着：用 PF-ODE 生成的样本与用反向 SDE 生成的样本来自同一个分布（但具体路径不同）。

### 7.5 实际使用中的 PF-ODE

将 score 用训练好的模型近似：

$$\mathrm{d}x = \left[f(x,t) - \frac{1}{2}g^2(t)\,s_\theta(x, t)\right]\mathrm{d}t$$

### 7.6 反向 SDE vs PF-ODE 对比

| | 反向 SDE | PF-ODE |
|---|---|---|
| **方程** | $\mathrm{d}x = [f - g^2 s_\theta]\mathrm{d}t + g\,\mathrm{d}\bar{W}$ | $\mathrm{d}x = [f - \frac{1}{2}g^2 s_\theta]\mathrm{d}t$ |
| **过程性质** | 随机 | 确定性 |
| **采样多样性** | 更高 | 更低 |
| **采样质量** | 更高 | 更低 |
| **采样速度** | 更慢 | 更快 |

> 📝 **注解：** 反向 SDE 的随机性带来了更高的多样性和质量，但也更慢。PF-ODE 的确定性使得可以使用更高效的 ODE 求解器，速度更快，但多样性和质量略有下降。实践中通常在速度优先时使用 PF-ODE。

### 7.7 与 DDIM 的联系

| | DDIM | PF-ODE |
|---|---|---|
| **性质** | DDPM 的确定性对应 | 反向 SDE 的确定性对应 |
| **能否复用训练好的模型？** | 是 | 是 |
| **采样多样性** | 较低 | 较低 |
| **采样质量** | 较低 | 较低 |
| **采样速度** | 更快 | 更快 |

> 📝 **注解：** DDIM 本质上就是 PF-ODE 的一种离散化！Lecture 1 中 DDIM 的确定性更新规则可以看作对 PF-ODE 的 Euler 离散化。这解释了为什么 DDIM 可以复用 DDPM 训练的模型——因为 PF-ODE 与反向 SDE 有相同的边际分布，所以用反向 SDE 训练的模型可以直接用于 PF-ODE 推理。

---

## 八、DPM-Solver：专为 PF-ODE 设计的求解器

### 8.1 复杂度度量：NFE

**NFE = Number of Function Evaluations**（函数评估次数）

每次函数评估 = 一次神经网络前向传播。NFE 越少，采样越快。

> 📝 **注解：** NFE 是衡量采样效率的标准指标。DDPM 需要 1000 NFE，DDIM 可以降到 10-50 NFE。DPM-Solver 的目标是在给定 NFE 预算下获得最佳质量。

### 8.2 传统 ODE 求解器

**Euler 方法：** 1 NFE/步，误差大

$$x_{t+\Delta t} = x_t + \Delta t \cdot v(x_t, t)$$

**Runge-Kutta 4 (RK4)：** 4 NFE/步，误差小

> 📝 **注解：** 传统求解器是"通用"的——它们不知道 PF-ODE 的特殊结构，因此效率不高。DPM-Solver 的关键洞察是：利用 PF-ODE 的特殊结构，设计专门的求解器。

### 8.3 PF-ODE 的特殊结构

> *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*, Lu et al., 2022.

PF-ODE 可以写成：

$$\frac{\mathrm{d}x}{\mathrm{d}t} = v(x, t)$$

利用扩散模型的特殊结构，$v$ 可以分解为：

- **关于 $x$ 线性**的部分（来自漂移项 $f(x,t)$）
- **关于 $x$ 非线性**的部分（来自 score 项 $s_\theta(x,t)$）

> 📝 **注解：** 这个分解是 DPM-Solver 效率的关键。线性部分可以精确求解，非线性部分用 Taylor 展开近似。这样，大部分计算量集中在非线性项（即神经网络评估）上，而线性部分是"免费"的。

### 8.4 DPM-Solver 的思路

**传统求解器：** 先离散化，再求解 → 通用但低效

**DPM-Solver：** 先精确求解线性部分，再对非线性部分离散化 → 专用但高效

> 📝 **注解：** DPM-Solver 使用"常数变易法"（variation of constants）和变量替换，将 PF-ODE 转化为一个关于 score 函数的积分方程。线性部分被精确积分，非线性部分（score）用 Taylor 展开近似。

### 8.5 DPM-Solver 变体

| 变体 | NFE | 精度 |
|------|-----|------|
| **DPM-Solver-1** | 1 NFE/步 | 一阶 Taylor 展开 |
| **DPM-Solver-2** | 2 NFE/步 | 二阶 Taylor 展开 |
| **DPM-Solver-k** | k NFE/步 | (k-1) 阶 Taylor 展开 |

> 📝 **注解：** DPM-Solver-k 使用 (k-1) 阶 Taylor 展开来近似 score 函数的积分。阶数越高，精度越好，但每步需要更多 NFE。在低 NFE 预算下（如 10-20 NFE），DPM-Solver-2 通常是最佳选择。

### 8.6 DPM-Solver 结论

**思路：**

```
前向 SDE → Fokker-Planck 方程 → 连续性方程 → PF-ODE
→ 利用 PF-ODE 的结构 → DPM-Solver
```

**性能：**
- 在给定 NFE 预算下，优于传统求解器
- 在低 NFE 区域优势最明显

**实际使用：**
- **无需重新训练**——直接使用训练好的 score/噪声预测模型
- 仅需 **10-20 NFE** 即可获得合理结果

> 📝 **注解：** DPM-Solver 是目前最流行的扩散模型快速采样方法之一。它被集成在 Diffusers、CompVis 等主流代码库中。与 DDIM 相比，DPM-Solver 在相同 NFE 下通常能获得更好的 FID 分数。

---

## 附录：关键概念速查表

| 概念 | 定义/公式 | 注解 |
|------|---------|------|
| **Score 函数** | $s(x) = \nabla_x \log p(x)$ | 对数概率的梯度，指向高密度方向 |
| **DSM 损失** | $\|s_\theta(z+\sigma\epsilon) + \epsilon/\sigma\|^2$ | 可处理的 score matching 损失 |
| **NCSN** | 多噪声水平条件 score 网络 | 解决单一噪声水平的局限 |
| **ALD** | 退火 Langevin 动力学 | NCSN 的采样方法 |
| **VP-SDE** | $\mathrm{d}x = -\frac{1}{2}\beta(t)x\,\mathrm{d}t + \sqrt{\beta(t)}\mathrm{d}W_t$ | DDPM 的连续形式 |
| **VE-SDE** | $\mathrm{d}x = \sqrt{\frac{\mathrm{d}\sigma^2}{\mathrm{d}t}}\mathrm{d}W_t$ | NCSN 的连续形式 |
| **反向 SDE** | $\mathrm{d}x = [f - g^2 s_\theta]\mathrm{d}t + g\,\mathrm{d}\bar{W}$ | Anderson (1982) 定理 |
| **PF-ODE** | $\mathrm{d}x = [f - \frac{1}{2}g^2 s_\theta]\mathrm{d}t$ | 与反向 SDE 同边际密度的 ODE |
| **DPM-Solver** | 利用 PF-ODE 结构的专用求解器 | 10-20 NFE 即可采样 |
| **NFE** | Number of Function Evaluations | 采样效率的标准度量 |
