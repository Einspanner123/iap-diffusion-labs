# 📚 MIT IAP 2026 · Lecture 1: Generative AI with Stochastic Differential Equations

## 流模型与扩散模型导论 —— 面向统计与概率论初学者

---

## 一、什么是生成式 AI？

生成式 AI 的核心能力是 **"创造"** ：给它一个提示（prompt），它能生成图片、视频、文本、甚至蛋白质结构。

**本课程聚焦的技术：Flow（流）模型 和 Diffusion（扩散）模型**——当前最先进的生成模型，驱动着 Stable Diffusion、DALL·E、OpenAI Sora、Meta MovieGen、AlphaFold3 等产品。

---

## 二、Section 1: 从"生成"到"采样"

### 2.1 我们要生成什么？——向量表示

一切生成对象都被表示为**高维向量**：

| 对象 | 向量表示 | 维度 |
|------|---------|------|
| 图片 | 高 $H$ × 宽 $W$ × 3 色彩通道 | $z \in \mathbb{R}^{H \times W \times 3}$ |
| 视频 | $T$ 帧 × 每帧是图片 | $z \in \mathbb{R}^{T \times H \times W \times 3}$ |
| 蛋白质 | $N$ 个原子 × 每个3个坐标 | $z \in \mathbb{R}^{N \times 3}$ |

> 统一记法：我们要生成的对象是一个 $d$ 维向量 $z \in \mathbb{R}^d$

### 2.2 什么才算"成功生成"？

提示："一张狗的图片"

| 噪声图 | 街景 | 猫 | 狗 |
|--------|------|-----|-----|
| 无用 < | 差 < | 错误动物 < | 很棒！|

评判标准其实是**"这张图在互联网上出现的可能性有多大"**：
- 噪声图 → 不可能出现
- 猫图 → 不太可能（给定 prompt 是"狗"）
- 真实狗图 → 非常可能

> 🎯 **关键洞察：图片质量 ≈ 它在数据分布下的概率高低**

### 2.3 数据分布 $p_{\text{data}}$

$$p_{\text{data}} : \mathbb{R}^d \to \mathbb{R}_{\geq 0}, \qquad z \mapsto p_{\text{data}}(z)$$

**解读：**
- $p_{\text{data}}$ 是一个**概率密度函数**
- 输入一个向量 $z$（比如一张图片的像素值），输出一个非负实数
- 输出值越大 = 这张图越"真实"

> ⚠️ 重要：我们**不知道** $p_{\text{data}}$ 的具体公式！我们只有从它采样得到的有限个样本（训练数据集）。

**生成 = 采样：**

$$z \sim p_{\text{data}}$$

> 从数据分布中"抽样"一个 $z$，这个 $z$ 就是生成的图片。

### 2.4 数据集

$$z_1, \ldots, z_N \sim p_{\text{data}}$$

数据集就是从数据分布中抽取的 $N$ 个有限样本。例如：
- 图片：互联网上公开的图片
- 视频：YouTube
- 蛋白质：Protein Data Bank

### 2.5 条件生成

除了无条件生成 $z \sim p_{\text{data}}$，还有**条件生成**：

$$z \sim p_{\text{data}}(\cdot | y)$$

其中 $y$ 是条件变量（如文本 prompt "Dog"、"Cat"、"Landscape"）。

> 本课先聚焦无条件生成，后面再学如何加入条件。

### 2.6 生成模型的本质

$$x \sim p_{\text{init}} \xrightarrow{\text{Generative Model}} z \sim p_{\text{data}}$$

- **输入**：从简单分布（如高斯分布 $p_{\text{init}} = \mathcal{N}(0, I_d)$）中采样的噪声
- **输出**：看起来像真实数据的样本

> 生成模型就是一台"噪声→数据"的变换机器。

---

## 三、Section 2: Flow 和 Diffusion 模型

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

> 💡 **直觉：** 粒子就像河流中的小船，在每个时刻、每个位置，按照向量场给出的速度向前走。

### 3.2 流（Flow）

给定向量场 $u_t$ 和初始点 $x_0$，ODE 的解 $\psi_t(x_0)$ 称为**流**（flow）：

$$\psi_t(x_0) = X_t \quad \text{（从 } x_0 \text{ 出发，沿 ODE 走到时刻 } t \text{ 的位置）}$$

**流映射**：一个函数，把初始位置映射到时刻 $t$ 的位置。

### 3.3 Picard-Lindelöf 定理（存在唯一性）

> **定理：** 如果向量场 $u_t(x)$ 连续可微且导数有界（更一般地，Lipschitz 连续），则 ODE 有**唯一解**。

**大白话：** 只要向量场（河流速度）足够"平滑"，小船的轨迹就是确定的、唯一的——不会分叉、不会消失。

### 3.4 例子：线性 ODE

$$u_t(x) = -\theta x \quad (\theta > 0)$$

解为：

$$\psi_t(x_0) = \exp(-\theta t) \cdot x_0$$

**验证：**
1. 初始条件：$\psi_0(x_0) = \exp(0) \cdot x_0 = x_0$ ✅
2. ODE 成立：$\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0) = -\theta \exp(-\theta t) x_0 = -\theta \psi_t(x_0) = u_t(\psi_t(x_0))$ ✅

> 直觉：粒子以指数速率衰减，趋向原点——像弹簧把物体拉回平衡位置。

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

> 💡 第①步：朝向量场方向走一小步

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

---

## 四、随机微分方程（SDE）与 Diffusion 模型

### 4.1 随机过程与布朗运动

**ODE 的轨迹是确定性的**——同一起点永远走出同一条路。

**随机过程**则不同：同一起点出发，每次走出的路径都不一样，因为路径中有**随机性**。

**布朗运动** $W_t$：
- 最基本的连续随机过程
- 就像一个**连续版的随机游走**（醉汉走路）
- 性质：$W_0 = 0$，增量 $W_{t+s} - W_t \sim \mathcal{N}(0, sI_d)$（正态分布，方差和时间成正比）

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

### 4.3 SDE 的存在唯一性

> **定理：** 如果向量场 $u_t(x)$ 连续可微、导数有界，且扩散系数 $\sigma_t$ 连续，则 SDE 有唯一解（分布意义上）。

**直觉理解（为什么需要这些条件？）：**

| 条件 | 作用 | 如果不满足会怎样？ |
|------|------|-------------------|
| $u_t(x)$ 连续可微 | 保证"推力方向"平滑变化 | 粒子可能突然跳变方向 |
| 导数有界 | 保证"推力大小"不会爆炸 | 粒子可能瞬间飞到无穷远 |
| $\sigma_t$ 连续 | 保证"噪声强度"平滑变化 | 随机性可能突然消失或爆炸 |

**类比：** 想象你在风中行走
- **漂移项** $u_t(x)$ = 风的方向和强度（需要平滑、不能无限大）
- **扩散项** $\sigma_t$ = 地面震动的大小（需要连续变化）
- **解的存在** = 你能走出一条路径
- **解的唯一** = 给定起点和风场，你的轨迹是确定的（分布意义上）

> 🎯 **一句话：** 只要"推力"和"噪声"都足够"规矩"（平滑、有界），SDE 就一定有解，而且解是唯一的。

### 4.4 例子：Ornstein-Uhlenbeck 过程

$$\mathrm{d}X_t = -\theta X_t \mathrm{d}t + \sigma \mathrm{d}W_t$$

- 漂移项 $-\theta X_t$：把粒子拉向原点（弹簧力）
- 扩散项 $\sigma \mathrm{d}W_t$：随机抖动

当 $\sigma$ 从小到大变化时：
- $\sigma \approx 0$：轨迹几乎和 ODE 一样光滑（指数衰减）
- $\sigma$ 很大：轨迹非常嘈杂（随机游走主导）

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

> 💡 第②步的含义：**确定性推力** $h \cdot u_t(X_t)$ + **随机抖动** $\sigma_t \sqrt{h} \cdot \epsilon$

**和 Euler 方法的对比：**

| | Euler (ODE) | Euler-Maruyama (SDE) |
|---|---|---|
| 更新公式 | $X_{t+h} = X_t + h \cdot u_t(X_t)$ | $X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t\sqrt{h}\,\epsilon$ |
| 多出来的 | 无 | $\sigma_t\sqrt{h}\,\epsilon$（随机噪声项） |

> 注意 $\sqrt{h}$：噪声的标准差和 $\sqrt{h}$ 成正比（不是 $h$），这是布朗运动的基本性质。

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

> 和 Flow Model 的唯一区别：每一步多了随机噪声。这在某些任务（如蛋白质生成）中可以提供更好的**多样性**。

---

## 五、总结与全局概览

### Section 1 的核心概念

| 概念 | 定义 |
|------|------|
| **生成对象** | 用向量 $z \in \mathbb{R}^d$ 表示 |
| **数据分布** $p_{\text{data}}$ | 对"好"的对象赋予高概率的分布 |
| **生成 = 采样** | 从 $p_{\text{data}}$ 中采样 |
| **数据集** | 从 $p_{\text{data}}$ 抽取的有限样本 |
| **条件生成** | 从 $p_{\text{data}}(\cdot\|y)$ 采样 |
| **生成模型** | 把简单分布（高斯）的样本变换为数据分布的样本 |

### Section 2 的核心概念

| 概念 | 公式 | 本质 |
|------|------|------|
| **ODE** | $\frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t)$ | 确定性轨迹 |
| **流（Flow）** | $\psi_t(x_0)$ | ODE 从 $x_0$ 出发的解 |
| **Euler 方法** | $X_{t+h} = X_t + h \cdot u_t(X_t)$ | 数值近似 ODE |
| **Flow Model** | 用神经网络 $u_t^\theta$ 做向量场 | 确定性生成 |
| **SDE** | $\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t$ | 带随机性的轨迹 |
| **布朗运动** $W_t$ | 连续随机游走 | SDE 中的噪声源 |
| **Euler-Maruyama** | $X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t\sqrt{h}\,\epsilon$ | 数值近似 SDE |
| **Diffusion Model** | 用神经网络 $u_t^\theta$ + 随机噪声 | 随机性生成 |

### 两种模型的关系

| 模型 | 方程 | 性质 |
|------|------|------|
| **Flow Model** | $\mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t$ | 确定性（ODE） |
| **Diffusion Model** | $\mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ | 随机性（SDE） |

> 当 $\sigma_t = 0$ 时，Diffusion Model 退化为 Flow Model。Flow Model 是 Diffusion Model 的特例。

**下一讲预告：** 如何训练这些模型？→ Flow Matching（流匹配）算法。
