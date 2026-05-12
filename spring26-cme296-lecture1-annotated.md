# [Spring 2026] CME 296 — Lecture 1: Generative AI with Stochastic Differential Equations

> **来源：** `spring26-cme296-lecture1.pdf`
> **课程：** Stanford CME 296, Spring 2026
> **日期：** 2026-04-03
> **注解版本：** 含详细数学推导与交叉引用

---

## 目录

1. [什么是生成式 AI](#一什么是生成式-ai)
2. [从"生成"到"采样"](#二从生成到采样)
3. [Flow 和 Diffusion 模型](#三flow-和-diffusion-模型)
4. [随机微分方程（SDE）与 Diffusion 模型](#四随机微分方程sde与-diffusion-模型)
5. [总结与全局概览](#五总结与全局概览)
6. [深度注解与补充](#六深度注解与补充)

---

## 一、什么是生成式 AI？

### 1.1 核心能力

生成式 AI 的核心能力是**"创造"**：给它一个提示（prompt），它能生成图片、视频、文本、甚至蛋白质结构。

**本课程聚焦的技术：Flow（流）模型和 Diffusion（扩散）模型**——当前最先进的生成模型，驱动着 Stable Diffusion、DALL·E、OpenAI Sora、Meta MovieGen、AlphaFold3 等产品。

### 1.2 本讲主线

这一讲回答三个核心问题：

1. 我们到底想生成什么对象？
2. 为什么"生成"可以理解成"从某个分布采样"？
3. 为什么 ODE/SDE 能把简单噪声逐步变成真实数据？

> **一句话总结：** 先把图片、视频、蛋白质都看成高维随机变量，再用连续时间动力系统把简单分布推到数据分布。

---

## 二、从"生成"到"采样"

### 2.1 我们要生成什么？——向量表示

一切生成对象都被表示为**高维向量**：

| 对象 | 向量表示 | 维度 |
|------|---------|------|
| 图片 | 高 $H$ × 宽 $W$ × 3 色彩通道 | $z \in \mathbb{R}^{H \times W \times 3}$ |
| 视频 | $T$ 帧 × 每帧是图片 | $z \in \mathbb{R}^{T \times H \times W \times 3}$ |
| 蛋白质 | $N$ 个原子 × 每个3个坐标 | $z \in \mathbb{R}^{N \times 3}$ |

> 📝 **注解：** 统一记法——我们要生成的对象是一个 $d$ 维向量 $z \in \mathbb{R}^d$。这个统一视角至关重要：无论生成什么，数学框架完全相同。例如一张 256×256 的 RGB 图片对应 $d = 256 \times 256 \times 3 = 196{,}608$ 维。

### 2.2 什么才算"成功生成"？

提示："一张狗的图片"

| 噪声图 | 街景 | 猫 | 狗 |
|--------|------|-----|-----|
| 无用 < | 差 < | 错误动物 < | 很棒！|

评判标准是**"这张图在互联网上出现的可能性有多大"**：
- 噪声图 → 不可能出现
- 猫图 → 不太可能（给定 prompt 是"狗"）
- 真实狗图 → 非常可能

> 🎯 **关键洞察：图片质量 ≈ 它在数据分布下的概率高低**

> 📝 **注解：** 这里的"可能性"不是主观判断，而是可以用概率密度函数 $p_{\text{data}}(z)$ 来量化。$p_{\text{data}}(z)$ 越大，说明这张图越"像"真实数据。这是整个生成模型框架的基石。

### 2.3 数据分布 $p_{\text{data}}$

$$p_{\text{data}} : \mathbb{R}^d \to \mathbb{R}_{\geq 0}, \qquad z \mapsto p_{\text{data}}(z)$$

**解读：**
- $p_{\text{data}}$ 是一个**概率密度函数**（Probability Density Function, PDF）
- 输入一个向量 $z$（比如一张图片的像素值），输出一个非负实数
- 输出值越大 = 这张图越"真实"
- 必须满足归一化条件：$\int_{\mathbb{R}^d} p_{\text{data}}(z)\,\mathrm{d}z = 1$

> ⚠️ **重要：** 我们**不知道** $p_{\text{data}}$ 的具体公式！我们只有从它采样得到的有限个样本（训练数据集）。这是生成模型面临的核心挑战——我们需要从有限样本中学习一个未知的分布。

**生成 = 采样：**

$$z \sim p_{\text{data}}$$

> 📝 **注解：** 符号 "$\sim$" 读作"服从……分布"。$z \sim p_{\text{data}}$ 意味着从数据分布中"抽样"一个 $z$，这个 $z$ 就是生成的图片。这个等式是生成模型的终极目标——构造一种机制，使得我们能够从 $p_{\text{data}}$ 中高效采样。

### 2.4 数据集

$$z_1, \ldots, z_N \sim p_{\text{data}}$$

数据集就是从数据分布中抽取的 $N$ 个有限样本。例如：
- 图片：互联网上公开的图片（如 LAION-5B 数据集包含 58.5 亿对图文）
- 视频：YouTube
- 蛋白质：Protein Data Bank

> 📝 **注解：** 数据集是 $p_{\text{data}}$ 的经验近似。根据大数定律，当 $N \to \infty$ 时，经验分布 $\frac{1}{N}\sum_{i=1}^N \delta(z - z_i)$ 会收敛到 $p_{\text{data}}$。但在实际中 $N$ 总是有限的，所以我们只能近似地学习 $p_{\text{data}}$。

### 2.5 条件生成

除了无条件生成 $z \sim p_{\text{data}}$，还有**条件生成**：

$$z \sim p_{\text{data}}(\cdot | y)$$

其中 $y$ 是条件变量（如文本 prompt "Dog"、"Cat"、"Landscape"）。

> 📝 **注解：** 条件生成是实际应用中最常见的场景。Stable Diffusion、DALL·E 等产品都是条件生成模型——用户输入文本描述 $y$，模型生成符合描述的图片 $z$。数学上，$p_{\text{data}}(\cdot|y)$ 是给定条件 $y$ 后的后验分布。本课先聚焦无条件生成，后面再学如何加入条件。

### 2.6 为什么起点通常选高斯分布？

生成模型并不是从"空白"开始生成，而是从一个容易采样的初始分布开始。最常见的选择是：

$$p_{\text{init}} = \mathcal{N}(0, I_d)$$

> 📝 **注解：** $\mathcal{N}(0, I_d)$ 表示 $d$ 维标准正态分布，其中 $0$ 是均值向量，$I_d$ 是 $d \times d$ 单位矩阵（协方差矩阵）。这意味着每个维度独立服从标准正态分布 $N(0,1)$。

选择高斯分布的原因：

| 原因 | 详细解释 |
|------|---------|
| **容易采样** | 每个维度独立采一个标准高斯随机数即可，复杂度 $O(d)$ |
| **数学上干净** | 旋转对称（各向同性），很多公式可解析求解 |
| **与后续训练兼容** | 高斯噪声与线性插值、score matching、SDE 理论都很好配合 |
| **中心极限定理** | 大量独立随机变量的和趋向高斯分布，具有普适性 |

> 📝 **深度注解：** 为什么不选均匀分布？均匀分布在高维空间中有"维度灾难"问题——大部分体积集中在边界附近，采样效率极低。高斯分布则没有这个问题，且其线性变换仍然是高斯分布（封闭性），这在后续推导中极为重要。

### 2.7 生成模型的本质

$$x \sim p_{\text{init}} \xrightarrow{\text{Generative Model}} z \sim p_{\text{data}}$$

- **输入**：从简单分布（如高斯分布 $p_{\text{init}} = \mathcal{N}(0, I_d)$）中采样的噪声
- **输出**：看起来像真实数据的样本

> 📝 **注解：** 生成模型就是一台"噪声→数据"的变换机器。数学上，这是一个从 $p_{\text{init}}$ 到 $p_{\text{data}}$ 的传输映射（transport map）。Flow/Diffusion 模型的独特之处在于：这个变换不是一步完成的，而是通过连续时间的动力系统逐步实现的。

---

## 三、Flow 和 Diffusion 模型

### 3.1 向量场与 ODE（常微分方程）

**向量场** $u_t(x)$：在每个位置 $x$、每个时刻 $t$，给出一个"速度方向"——就像风场图上每个点的风向和风速。

**ODE（常微分方程）：**

$$X_0 = x_0, \qquad \frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t)$$

**逐项翻译：**

| 符号 | 含义 | 比喻 |
|------|------|------|
| $X_0 = x_0$ | 初始条件：粒子在 $t=0$ 的位置 | 小船的出发点 |
| $\frac{\mathrm{d}}{\mathrm{d}t}X_t$ | 粒子在时刻 $t$ 的速度（位置的变化率） | 小船此刻的速度 |
| $u_t(X_t)$ | 向量场在粒子当前位置给出的速度 | 河流在小船位置的流速 |

> 📝 **注解：** ODE 是确定性动力系统的基础。$\frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t)$ 也可以写成积分形式 $X_t = x_0 + \int_0^t u_s(X_s)\,\mathrm{d}s$。这两种形式是等价的，微分形式更直观，积分形式更适合数值计算。

> 💡 **直觉：** 粒子就像河流中的小船，在每个时刻、每个位置，按照向量场给出的速度向前走。

### 3.2 流（Flow）

给定向量场 $u_t$ 和初始点 $x_0$，ODE 的解 $\psi_t(x_0)$ 称为**流**（flow）：

$$\psi_t(x_0) = X_t \quad \text{（从 } x_0 \text{ 出发，沿 ODE 走到时刻 } t \text{ 的位置）}$$

**流映射**：一个函数，把初始位置映射到时刻 $t$ 的位置。

> 📝 **注解：** 流 $\psi_t$ 有以下重要性质：
> 1. **初始条件：** $\psi_0(x_0) = x_0$（$t=0$ 时粒子还在起点）
> 2. **半群性质：** $\psi_{t+s}(x_0) = \psi_t(\psi_s(x_0))$（先走 $s$ 步再走 $t$ 步 = 直接走 $t+s$ 步）
> 3. **可逆性（当向量场足够好时）：** $\psi_t^{-1}$ 存在，可以从终点反推起点
>
> 性质 2 在实践中意味着：我们可以用任意步长来数值求解 ODE，结果是一致的（在精确求解的情况下）。

### 3.3 Picard-Lindelöf 定理（存在唯一性）

> **定理：** 如果向量场 $u_t(x)$ 连续可微且导数有界（更一般地，Lipschitz 连续），则 ODE 有**唯一解**。

> 📝 **注解：** Lipschitz 连续的条件是：存在常数 $L > 0$，使得对所有 $x, y$ 和 $t$，有 $\|u_t(x) - u_t(y)\| \leq L\|x - y\|$。这比"连续可微"稍弱，但足以保证唯一性。在神经网络中，Lipschitz 连续性可以通过谱归一化（spectral normalization）等技术来近似保证。

**大白话：** 只要向量场（河流速度）足够"平滑"，小船的轨迹就是确定的、唯一的——不会分叉、不会消失。

> 📝 **为什么需要这个定理？** 因为 Flow Model 的核心假设是：给定向量场，从噪声出发的轨迹是唯一确定的。如果解不唯一，那么同一个噪声可能生成不同的图片，这会导致生成结果不可控。Picard-Lindelöf 定理保证了这种情况不会发生。

### 3.4 例子：线性 ODE

$$u_t(x) = -\theta x \quad (\theta > 0)$$

解为：

$$\psi_t(x_0) = \exp(-\theta t) \cdot x_0$$

**验证：**
1. 初始条件：$\psi_0(x_0) = \exp(0) \cdot x_0 = x_0$ ✅
2. ODE 成立：$\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0) = -\theta \exp(-\theta t) x_0 = -\theta \psi_t(x_0) = u_t(\psi_t(x_0))$ ✅

> 📝 **注解：** 这个例子展示了 ODE 的解析求解过程。对于线性向量场 $u_t(x) = Ax$（$A$ 是矩阵），解为 $\psi_t(x_0) = \exp(At) \cdot x_0$，其中 $\exp(At)$ 是矩阵指数。这是线性系统理论的基础结果。在实际的 Flow Model 中，向量场是非线性的（由神经网络参数化），所以通常没有解析解，需要数值方法。

### 3.5 Euler 方法——数值求解 ODE

当 ODE 没有解析解时，用 **Euler 方法**近似求解：

**算法：Euler 方法**

| 步骤 | 操作 |
|------|------|
| **输入** | 向量场 $u_t$，初始条件 $x_0$，步数 $n$ |
| **初始化** | $t \gets 0$，步长 $h \gets 1/n$，$X_0 \gets x_0$ |
| **循环** | 对 $i = 1, \ldots, n-1$： |
| ① | $X_{t+h} \gets X_t + h \cdot u_t(X_t)$ |
| ② | $t \gets t + h$ |
| **输出** | 轨迹 $X_0, X_h, X_{2h}, \ldots, X_1$ |

> 📝 **注解：** Euler 方法是一阶方法，局部截断误差为 $O(h^2)$，全局误差为 $O(h)$。这意味着步长减半，误差大约减半。更高级的方法（如 Runge-Kutta 4阶方法）可以达到 $O(h^4)$ 的全局误差，但每步需要更多计算。在 Flow Model 的实践中，Euler 方法因其简单性和并行性而被广泛使用，通常配合 20-1000 步来平衡精度和速度。

**步长的权衡：**
- **大步长**：速度快，但误差大（像用直尺画曲线）
- **小步长**：误差小，但计算量大

### 3.6 Flow Model（流模型）的采样算法

把向量场 $u_t$ 换成**神经网络** $u_t^\theta$，就得到了 Flow Model 的采样算法：

**算法：Flow Model 采样**

| 步骤 | 操作 |
|------|------|
| **输入** | 神经网络向量场 $u_t^\theta$，步数 $n$ |
| **初始化** | $t \gets 0$，步长 $h \gets 1/n$ |
| **采样起点** | $X_0 \sim p_{\text{init}}$（随机噪声） |
| **循环** | 对 $i = 1, \ldots, n-1$： |
| ① | $X_{t+h} \gets X_t + h \cdot u_t^\theta(X_t)$ |
| ② | $t \gets t + h$ |
| **输出** | $X_1$（生成的样本） |

> 🎯 从噪声出发，沿着神经网络学到的"河流方向"走，走到终点就是一张生成的图片！

> 📝 **注解：** 注意这里 $u_t^\theta(X_t)$ 的输入包含时间 $t$ 和位置 $X_t$。神经网络需要同时感知"现在是什么时刻"和"粒子在哪里"，才能给出正确的速度方向。在实践中，时间 $t$ 通常通过正弦位置编码（sinusoidal positional encoding）注入网络。

### 3.7 为什么 ODE 适合做生成？

ODE 适合做生成模型，核心是它定义了一个**确定性的连续变换**：

- 起点样本 $X_0$ 一旦给定，终点 $X_1$ 就确定
- 如果向量场足够平滑，这个变换通常是可逆且稳定的
- 因而可以把"简单分布"连续地推送成"复杂分布"

> 📝 **注解：** ODE 变换的"分布推送"效果可以通过**变量替换公式**来理解。如果 $X_1 = \psi_1(X_0)$，那么 $p_1(x) = p_0(\psi_1^{-1}(x)) \cdot |\det \nabla \psi_1^{-1}(x)|$。这意味着 ODE 变换不仅移动了粒子的位置，还改变了概率密度的"浓度"——某些区域概率变高，某些变低。这正是从简单分布到复杂分布的关键。

---

## 四、随机微分方程（SDE）与 Diffusion 模型

### 4.1 随机过程与布朗运动

**ODE 的轨迹是确定性的**——同一起点永远走出同一条路。

**随机过程**则不同：同一起点出发，每次走出的路径都不一样，因为路径中有**随机性**。

**布朗运动** $W_t$：
- 最基本的连续随机过程
- 就像一个**连续版的随机游走**（醉汉走路）
- 性质：$W_0 = 0$，增量 $W_{t+s} - W_t \sim \mathcal{N}(0, sI_d)$（正态分布，方差和时间成正比）

> 📝 **注解：** 布朗运动（又称 Wiener 过程）有以下关键性质：
> 1. **独立增量：** $W_{t_2} - W_{t_1}$ 与 $W_{t_1} - W_{t_0}$ 独立（不重叠区间的增量互不影响）
> 2. **正态增量：** $W_{t+s} - W_t \sim \mathcal{N}(0, sI_d)$（增量的方差与时间长度成正比）
> 3. **连续路径：** $W_t$ 的轨迹几乎必然连续（但处处不可微——这是它最反直觉的性质）
> 4. **不可微性：** $\frac{\mathrm{d}W_t}{\mathrm{d}t}$ 不存在！这就是为什么 SDE 写成 $\mathrm{d}W_t$ 而不是 $\frac{\mathrm{d}W_t}{\mathrm{d}t}\mathrm{d}t$
>
> 性质 4 是 SDE 与 ODE 在数学处理上最大的不同。SDE 不能用普通的微积分，必须用 Itô 随机微积分。

### 4.2 SDE（随机微分方程）

$$\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$

**和 ODE 对比：**

| | ODE | SDE |
|---|---|---|
| 公式 | $\mathrm{d}X_t = u_t(X_t)\mathrm{d}t$ | $\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t$ |
| 多出来的项 | 无 | $\sigma_t \mathrm{d}W_t$（随机噪声） |
| 轨迹 | 确定的 | 随机的 |
| 比喻 | 平静河流中的小船 | 风浪中的小船 |

- $u_t(X_t)\mathrm{d}t$：**漂移项**（drift）——确定性的推力
- $\sigma_t \mathrm{d}W_t$：**扩散项**（diffusion）——随机的抖动
- $\sigma_t$：**扩散系数**——控制随机性的大小

> 📝 **注解：** SDE 的严格数学含义是积分方程：$X_t = X_0 + \int_0^t u_s(X_s)\,\mathrm{d}s + \int_0^t \sigma_s\,\mathrm{d}W_s$。第一个积分是普通的 Riemann 积分，第二个积分是 Itô 随机积分。Itô 积分与 Stratonovich 积分是两种不同的定义方式，它们在处理 $\mathrm{d}W_t$ 的"取值时刻"上有区别。在扩散模型文献中通常使用 Itô 积分。

### 4.3 SDE 的存在唯一性

> **定理：** 如果向量场 $u_t(x)$ 连续可微、导数有界，且扩散系数 $\sigma_t$ 连续，则 SDE 有唯一解（分布意义上）。

**直觉理解（为什么需要这些条件？）：**

| 条件 | 作用 | 如果不满足会怎样？ |
|------|------|-------------------|
| $u_t(x)$ 连续可微 | 保证"推力方向"平滑变化 | 粒子可能突然跳变方向 |
| 导数有界 | 保证"推力大小"不会爆炸 | 粒子可能瞬间飞到无穷远 |
| $\sigma_t$ 连续 | 保证"噪声强度"平滑变化 | 随机性可能突然消失或爆炸 |

> 📝 **注解：** 注意 SDE 的"唯一性"是**分布意义上**的，不是路径意义上的。也就是说，给定相同的初始条件，不同次运行会走出不同的路径（因为有随机性），但这些路径的**统计分布**是唯一确定的。这与 ODE 的路径唯一性形成对比。

### 4.4 例子：Ornstein-Uhlenbeck 过程

$$\mathrm{d}X_t = -\theta X_t \mathrm{d}t + \sigma \mathrm{d}W_t$$

- 漂移项 $-\theta X_t$：把粒子拉向原点（弹簧力/均值回复力）
- 扩散项 $\sigma \mathrm{d}W_t$：随机抖动

当 $\sigma$ 从小到大变化时：
- $\sigma \approx 0$：轨迹几乎和 ODE 一样光滑（指数衰减）
- $\sigma$ 很大：轨迹非常嘈杂（随机游走主导）

> 📝 **注解：** OU 过程是 SDE 理论中最重要的例子之一。它的解析解为 $X_t = e^{-\theta t}X_0 + \sigma\int_0^t e^{-\theta(t-s)}\,\mathrm{d}W_s$，稳态分布为 $\mathcal{N}(0, \frac{\sigma^2}{2\theta})$。在扩散模型中，前向加噪过程（VP-SDE）就是一个 OU 过程。详见 [ScoreSDE_Detailed.md](notes/ScoreSDE_Detailed.md)。

### 4.5 Euler-Maruyama 方法——数值求解 SDE

**算法：Euler-Maruyama 方法**

| 步骤 | 操作 |
|------|------|
| **输入** | 向量场 $u_t$，扩散系数 $\sigma_t$，初始条件 $x_0$，步数 $n$ |
| **初始化** | $t \gets 0$，步长 $h \gets 1/n$，$X_0 \gets x_0$ |
| **循环** | 对 $i = 1, \ldots, n-1$： |
| ① | 采样 $\epsilon \sim \mathcal{N}(0, I_d)$ |
| ② | $X_{t+h} \gets X_t + h \cdot u_t(X_t) + \sigma_t \cdot \sqrt{h} \cdot \epsilon$ |
| ③ | $t \gets t + h$ |
| **输出** | 轨迹 $X_0, X_h, X_{2h}, \ldots, X_1$ |

> 📝 **注解：** 为什么噪声项是 $\sigma_t \sqrt{h} \cdot \epsilon$ 而不是 $\sigma_t \cdot h \cdot \epsilon$？这是因为布朗运动的增量 $W_{t+h} - W_t \sim \mathcal{N}(0, h \cdot I_d)$，其标准差为 $\sqrt{h}$（不是 $h$）。这是布朗运动最根本的数学性质——方差与时间成正比，标准差与时间的平方根成正比。直观理解：随机游走的"扩散距离"与时间的平方根成正比，而非线性正比。

**和 Euler 方法的对比：**

| | Euler (ODE) | Euler-Maruyama (SDE) |
|---|---|---|
| 更新公式 | $X_{t+h} = X_t + h \cdot u_t(X_t)$ | $X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t\sqrt{h}\,\epsilon$ |
| 多出来的 | 无 | $\sigma_t\sqrt{h}\,\epsilon$（随机噪声项） |
| 收敛阶 | $O(h)$（强收敛） | $O(\sqrt{h})$（强收敛） |

> 📝 **注解：** SDE 数值方法的收敛速度比 ODE 慢——Euler-Maruyama 的强收敛阶是 $O(\sqrt{h})$，而 Euler 方法的强收敛阶是 $O(h)$。这意味着 SDE 需要更多的步数来达到相同的精度。这也是为什么 Diffusion Model 的采样通常需要更多步数（如 1000 步）而 Flow Model 可以用更少步数（如 20 步）。

### 4.6 Diffusion Model 的采样算法

**算法：Diffusion Model 采样**

| 步骤 | 操作 |
|------|------|
| **输入** | 神经网络向量场 $u_t^\theta$，扩散系数 $\sigma_t$，步数 $n$ |
| **初始化** | $t \gets 0$，步长 $h \gets 1/n$ |
| **采样起点** | $X_0 \sim p_{\text{init}}$（随机噪声） |
| **循环** | 对 $i = 1, \ldots, n-1$： |
| ① | 采样 $\epsilon \sim \mathcal{N}(0, I_d)$ |
| ② | $X_{t+h} \gets X_t + h \cdot u_t^\theta(X_t) + \sigma_t \cdot \sqrt{h} \cdot \epsilon$ |
| ③ | $t \gets t + h$ |
| **输出** | $X_1$（生成的样本） |

> 📝 **注解：** 和 Flow Model 的唯一区别是每一步多了随机噪声 $\sigma_t\sqrt{h}\,\epsilon$。这个随机性在某些任务中是有益的：
> - **多样性：** 同一个噪声起点，不同的随机采样路径会生成不同的图片
> - **探索性：** 在蛋白质生成等需要探索构象空间的任务中，随机性有助于发现新的结构
> - **鲁棒性：** 随机性可以防止模型"过拟合"到某条特定路径
>
> 但随机性也有代价：采样结果不完全可控，且需要更多步数来保证质量。

---

## 五、总结与全局概览

### Section 1 的核心概念

| 概念 | 定义 | 注解 |
|------|------|------|
| **生成对象** | 用向量 $z \in \mathbb{R}^d$ 表示 | 统一了图片、视频、蛋白质等 |
| **数据分布** $p_{\text{data}}$ | 对"好"的对象赋予高概率的分布 | 未知，只能通过数据集近似 |
| **生成 = 采样** | 从 $p_{\text{data}}$ 中采样 | 生成模型的终极目标 |
| **数据集** | 从 $p_{\text{data}}$ 抽取的有限样本 | 经验近似 |
| **条件生成** | 从 $p_{\text{data}}(\cdot\|y)$ 采样 | 实际应用最常见 |
| **生成模型** | 把简单分布的样本变换为数据分布的样本 | 噪声→数据的变换机器 |

### Section 2 的核心概念

| 概念 | 公式 | 本质 | 注解 |
|------|------|------|------|
| **ODE** | $\frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t)$ | 确定性轨迹 | Flow Model 的数学基础 |
| **流（Flow）** | $\psi_t(x_0)$ | ODE 从 $x_0$ 出发的解 | 具有半群性质和可逆性 |
| **Euler 方法** | $X_{t+h} = X_t + h \cdot u_t(X_t)$ | 数值近似 ODE | 一阶方法，$O(h)$ 收敛 |
| **Flow Model** | 用神经网络 $u_t^\theta$ 做向量场 | 确定性生成 | 少步数即可采样 |
| **SDE** | $\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t$ | 带随机性的轨迹 | Diffusion Model 的数学基础 |
| **布朗运动** $W_t$ | 连续随机游走 | SDE 中的噪声源 | 增量方差与时间成正比 |
| **Euler-Maruyama** | $X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t\sqrt{h}\,\epsilon$ | 数值近似 SDE | $O(\sqrt{h})$ 强收敛 |
| **Diffusion Model** | 用神经网络 $u_t^\theta$ + 随机噪声 | 随机性生成 | 更多多样性，需更多步数 |

### 两种模型的关系

| 模型 | 方程 | 性质 |
|------|------|------|
| **Flow Model** | $\mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t$ | 确定性（ODE） |
| **Diffusion Model** | $\mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ | 随机性（SDE） |

> 当 $\sigma_t = 0$ 时，Diffusion Model 退化为 Flow Model。Flow Model 是 Diffusion Model 的特例。

> 📝 **注解：** 这个关系非常重要——它意味着 Flow Model 和 Diffusion Model 不是两种完全不同的方法，而是同一个连续谱上的两个端点。在实践中，可以通过调节 $\sigma_t$ 的大小来在确定性和随机性之间权衡。这也解释了为什么 Flow Matching（Lecture 2）可以作为统一的训练框架同时适用于两种模型。

---

## 六、深度注解与补充

### 6.1 从概率论视角理解 ODE 的"分布推送"

ODE 不仅能移动单个粒子，还能**推送整个概率分布**。如果 $X_0 \sim p_0$，那么 $X_t = \psi_t(X_0)$ 的分布 $p_t$ 满足**连续性方程**（Continuity Equation）：

$$\frac{\partial p_t}{\partial t} + \text{div}(p_t \cdot u_t) = 0$$

> 📝 **注解：** 连续性方程来自物理学中的质量守恒定律。$\text{div}(p_t \cdot u_t)$ 是概率流 $p_t \cdot u_t$ 的散度，描述了概率"流入"和"流出"某个区域的净量。如果散度为正，说明该区域的概率在减少（概率流出）；如果散度为负，概率在增加（概率流入）。这个方程保证了总概率始终为 1。

### 6.2 Fokker-Planck 方程——SDE 的分布演化

对于 SDE $\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t$，其分布 $p_t$ 满足 **Fokker-Planck 方程**：

$$\frac{\partial p_t}{\partial t} = -\text{div}(p_t \cdot u_t) + \frac{1}{2}\sigma_t^2 \Delta p_t$$

其中 $\Delta = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2}$ 是 Laplacian 算子。

> 📝 **注解：** 与连续性方程相比，Fokker-Planck 方程多了一项 $\frac{1}{2}\sigma_t^2 \Delta p_t$，这是扩散项。它的物理含义是：随机噪声使得概率分布"扩散"——从高浓度区域向低浓度区域扩散。当 $\sigma_t = 0$ 时，Fokker-Planck 方程退化为连续性方程。详见 [ScoreSDE_Detailed.md](notes/ScoreSDE_Detailed.md)。

### 6.3 为什么高维高斯采样是"容易的"？

在 $d$ 维空间中，从 $\mathcal{N}(0, I_d)$ 采样的复杂度仅为 $O(d)$——每个维度独立采一个标准正态随机数即可。但直接从 $p_{\text{data}}$ 采样通常是不可行的，因为 $p_{\text{data}}$ 是一个极其复杂的、未知的、高维的概率分布。

> 📝 **注解：** 这就是为什么生成模型需要设计巧妙的采样策略。Flow/Diffusion 模型的核心思路是：不直接从 $p_{\text{data}}$ 采样，而是构造一条从 $p_{\text{init}}$（容易采样）到 $p_{\text{data}}$（难采样）的"路径"，然后沿着这条路径逐步变换样本。这条"路径"就是 ODE/SDE 定义的连续变换。

### 6.4 与 VAE 的联系

变分自编码器（VAE）是另一种重要的生成模型，它也使用了"简单分布→复杂分布"的思路，但方法不同：

| | VAE | Flow/Diffusion |
|---|---|---|
| 变换方式 | 编码器一步映射 | ODE/SDE 逐步变换 |
| 潜变量 | 低维瓶颈 | 同维（$d$ 维噪声→$d$ 维数据） |
| 训练目标 | ELBO（证据下界） | 向量场回归 |
| 生成质量 | 中等 | 极高 |

> 📝 **注解：** VAE 的编码器将数据压缩到低维潜空间，再由解码器从潜空间生成数据。这种"瓶颈"结构限制了生成质量。Flow/Diffusion 模型没有瓶颈——噪声和数据的维度相同，变换是同维的。详见 [VAE_Tutorial.md](notes/VAE_Tutorial.md)。

### 6.5 扩散模型的历史脉络

```
VAE (2013) → GAN (2014) → Flow (2018) → DDPM (2020) → Score SDE (2021) → DDIM (2021) → Flow Matching (2023)
```

> 📝 **注解：** 
> - **DDPM** 首次让扩散模型成为主流，但采样慢（1000步）→ 详见 [DDPM_Detailed.md](notes/DDPM_Detailed.md)
> - **Score SDE** 将 DDPM 统一到 SDE 框架，揭示了与 ODE 的联系 → 详见 [ScoreSDE_Detailed.md](notes/ScoreSDE_Detailed.md)
> - **DDIM** 发现可以去掉随机性，用 ODE 采样，速度提升 20 倍 → 详见 [DDIM_Detailed.md](notes/DDIM_Detailed.md)
> - **Flow Matching** 提出更简洁的训练框架，是当前最前沿的方法 → 详见 [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md)

### 6.6 复习时只记这四句话

1. 生成对象统一看成高维随机变量 $z \in \mathbb{R}^d$。
2. 生成任务本质上是从数据分布 $p_{\text{data}}$ 采样。
3. Flow model 用 ODE 把噪声连续推向数据，轨迹是确定的。
4. Diffusion model 用 SDE 做同样的事，只是中间额外加入随机扰动。

---

> **下一讲预告：** 如何训练这些模型？→ Flow Matching（流匹配）算法。详见 [Lecture 2 注解文档](spring26-cme296-lecture2-annotated.md)。
