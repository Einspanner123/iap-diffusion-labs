# [Spring 2026] CME 296 — Lecture 3: Flow Matching

> **来源：** `spring26-cme296-lecture3.pdf`（共 96 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [上讲回顾](#一上讲回顾)
2. [问题建模与直觉](#二问题建模与直觉)
3. [Flow Matching](#三flow-matching)
4. [训练](#四训练)
5. [推理](#五推理)
6. [Rectified Flow](#六rectified-flow)
7. [讨论：范式对比](#七讨论范式对比)

---

## 一、上讲回顾

### 1.1 Lecture 1 与 Lecture 2 的回顾

**Lecture 1（DDPM）：** 预测噪声 $\epsilon$ 来去噪

**Lecture 2（Score Matching / DSM）：** 预测 score $s = \nabla_x \log p(x)$

| | DDPM | NCSN |
|---|---|---|
| **预测目标** | 从噪声样本中去除的噪声 | 从噪声到干净的 score |

### 1.2 SDE 统一视角

前向与反向 SDE：

| | 前向 SDE | 反向 SDE |
|---|---|---|
| **漂移** | $f(x,t)$ | $f(x,t) - g^2(t)\nabla_x\log p_t(x)$ |
| **扩散** | $g(t)$ | $g(t)$ |

> 📝 **注解：** Lecture 1 和 2 从"噪声预测"和"score 预测"两个角度建立了扩散模型。本讲将介绍第三种范式——**Flow Matching**，它预测的是**速度场（velocity field）**，即样本应该往哪个方向、以多快的速度移动。

---

## 二、问题建模与直觉

### 2.1 本讲主题

**Flow Matching**——用速度场（velocity）将初始分布传输到目标分布。

### 2.2 Flow 的直觉

**想法：** 将概率质量从初始分布"运输"到目标分布。

想象初始分布是噪声，目标分布是数据。Flow 就像一条河流，将噪声"搬运"到数据。

> 📝 **注解：** Flow 的核心思想是**连续变换**——通过一个连续的速度场，将一个分布平滑地变形为另一个分布。与扩散模型不同，Flow Matching 从一开始就是确定性的 ODE，不需要随机项。

### 2.3 重要约定：时间方向

⚠️ **注意：** Flow Matching 的时间方向与 Lecture 1/2 **相反**！

| | Lecture 1 & 2（扩散/Score） | Lecture 3（Flow Matching） |
|---|---|---|
| $t=0$ | 干净图像 | 噪声（初始分布） |
| $t=1$ | 噪声 | 干净图像（目标分布） |

> 📝 **注解：** 这是一个常见的混淆点。在扩散模型中，时间从数据到噪声；在 Flow Matching 中，时间从噪声到数据。两种约定各有道理——扩散模型强调"加噪过程"，Flow Matching 强调"生成过程"。

### 2.4 基本记号

| 记号 | 含义 |
|------|------|
| $x_t$ | 时刻 $t$ 的观测值（轨迹） |
| 轨迹（Trajectory） | 单个观测随时间 $t$ 的路径 |
| 流（Flow） | 从不同 $x_0$ 出发的轨迹集合 |
| 概率路径（Probability path） $p_t(x)$ | 时刻 $t$ 时 $x_t$ 的概率分布 |
| 向量场/速度（Vector field / velocity）$v_t(x)$ | 在时刻 $t$、位置 $x$ 处的移动方向和速度 |

### 2.5 速度 vs Score 函数

| | 速度（Velocity） | Score 函数 |
|---|---|---|
| **比喻** | 🛣 高速公路——告诉你往哪走、走多快 | 🧭 指南针——只告诉你方向 |
| **信息量** | 方向 + 速度 | 仅方向 |
| **依赖** | 当前位置和时间 | 当前位置和时间 |

> 📝 **注解：** 速度场比 score 函数信息更丰富——score 只告诉你"上坡方向"，而速度告诉你"往哪走、走多快"。这使得 Flow Matching 的 ODE 求解更高效。

### 2.6 单样本视角：ODE

从单个样本的角度，轨迹 $x_t$ 满足常微分方程（ODE）：

$$\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_t(x_t)$$

**条件：** 如果速度场 $v_t$ 是 Lipschitz 连续的，则轨迹唯一。

> 📝 **注解：** Lipschitz 连续性保证了 ODE 解的存在唯一性（Picard-Lindelöf 定理）。这在 Flow Matching 中很重要——它保证了从同一初始点出发，轨迹是唯一确定的。

### 2.7 分布视角：质量守恒

从分布的角度，概率密度的时间演化满足**连续性方程**：

$$\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t \cdot v_t) = 0$$

- 左边第一项：某位置处概率密度的时间变化
- 左边第二项：该位置的"流入"减"流出"

> 📝 **注解：** 连续性方程是质量守恒的数学表述——概率密度不会凭空产生或消失，只会从一个地方"流"到另一个地方。这与 Lecture 2 中的 Fokker-Planck 方程类似，但没有扩散项（因为 ODE 没有随机性）。

### 2.8 散度的直觉（1D）

**散度** $\nabla_x \cdot v$ 衡量向量场的"发散程度"：

- **正散度**：流出多于流入 → 密度降低
- **负散度**：流入多于流出 → 密度升高

推广到 $d$ 维：$\nabla_x \cdot v = \sum_{i=1}^d \frac{\partial v_i}{\partial x_i}$

### 2.9 散度应该对谁取？

**问题：** 应该对哪个向量场取散度？

- ❌ 对向量场 $v_t$ 本身取散度
- ✅ 对**概率通量** $p_t \cdot v_t$ 取散度

> 📝 **注解：** 连续性方程中散度的对象是概率通量 $p_t v_t$，而不是速度场 $v_t$ 本身。这是因为我们关心的是概率密度的流动，而不仅仅是速度场。概率密度高的地方，即使速度小，也可能有大量的概率"流过"。

### 2.10 连续性方程

$$\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot (p_t(x) \cdot v_t(x))$$

> 📝 **注解：** 这就是连续性方程的标准形式。它建立了速度场 $v_t$ 和概率路径 $p_t$ 之间的关系。如果知道了 $v_t$，就可以通过连续性方程确定 $p_t$ 的演化；反之亦然。

### 2.11 两个视角的统一

| 视角 | 方程 | 含义 |
|------|------|------|
| **单样本** | $\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_t(x_t)$ | 速度场"生成"轨迹 |
| **分布** | $\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t v_t) = 0$ | 速度场"生成"概率路径 |

> 📝 **注解：** 这两个视角是等价的——单样本的 ODE 积分起来就是分布的连续性方程。Flow Matching 的核心思路是：学习速度场 $v_t$，使得它"生成"的概率路径从 $p_0$（噪声）变为 $p_1$（数据）。

### 2.12 Flow 模型的策略

**目标：** 将 $p_0$（初始分布，噪声）映射到 $p_1$（目标分布，数据）

**策略：**

1. **训练：** 估计速度场 $v_t(x)$，使得对所有时间和位置，连续性方程成立
2. **推理：** 从初始分布采样，用学到的速度场数值求解 ODE

> 📝 **注解：** 与扩散模型类似，Flow Matching 的训练目标是学习一个向量场。但与扩散模型不同的是，Flow Matching 的向量场直接描述了样本的移动方向，而不是噪声或 score。

### 2.13 早期尝试：Neural ODE

> *Neural Ordinary Differential Equations*, Chen et al., 2018.

**想法：** 将连续性方程转化为似然最大化问题，在训练时模拟 ODE。

**局限：** 训练时需要模拟 ODE → 🐌 训练慢且昂贵！

> 📝 **注解：** Neural ODE 的训练需要在每次前向传播时求解 ODE（通常用 adjoint method），这非常耗时。Flow Matching 的关键创新是：**不需要在训练时模拟 ODE**——通过条件概率路径的技巧，可以直接计算损失函数。

---

## 三、Flow Matching

### 3.1 Flow Matching 的目标

**FM = Flow Matching**

**目标：** 用神经网络 $v_\theta(x, t)$ 估计速度场 $v_t(x)$。

**问题：** 我们无法直接计算 $v_t(x)$。

### 3.2 回到向量场的定义

向量场 $v_t(x)$ 是未知的——我们不知道从噪声到数据的"正确"速度场应该长什么样。

### 3.3 退回到更简单的设定

**想法：** 先考虑一个更简单的情况——目标分布是单个数据点（Dirac 分布）。

### 3.4 如何从初始分布到 Dirac 分布？

**问题：** 如何定义从噪声 $p_0$ 到单个数据点 $x_1$ 的概率路径和速度场？

### 3.5 条件概率路径

**提议：** 条件高斯概率路径

$$p_t(x|x_1) = \mathcal{N}(x \mid \mu_t(x_1),\, \sigma_t^2(x_1) I)$$

其中 $\mu_t(x_1)$ 是条件均值，$\sigma_t^2(x_1)$ 是条件方差。

> 📝 **注解：** 条件概率路径描述了"给定目标数据点 $x_1$，时刻 $t$ 的分布应该长什么样"。在 $t=0$ 时，它应该是噪声分布；在 $t=1$ 时，它应该集中在 $x_1$ 附近。

### 3.6 条件向量场

给定条件概率路径，可以通过连续性方程推导出对应的条件向量场 $v_t(x|x_1)$。

### 3.7 条件向量场生成条件概率路径

**关键结论：** 条件向量场 $v_t(x|x_1)$ **生成**条件概率路径 $p_t(x|x_1)$。

即：如果 $p_t(x|x_1)$ 满足连续性方程 $\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t \cdot v_t) = 0$，则 $v_t(x|x_1)$ 就是生成该路径的速度场。

**推论：** 如果 $p_0(x|x_1) = p_0(x)$（初始分布与条件无关）且 $p_1(x|x_1) \approx \delta(x - x_1)$（终态集中在数据点），则条件向量场将噪声传输到数据点。

> 📝 **注解：** 这是 Flow Matching 的基础——我们不需要知道边际向量场 $v_t(x)$，只需要知道条件向量场 $v_t(x|x_1)$，因为它有简单的解析表达式。

### 3.8 边际概率路径

**边际概率路径**是条件概率路径的聚合：

$$p_t(x) = \int p_t(x|x_1)\, p_{\text{data}}(x_1)\,\mathrm{d}x_1$$

- $p_0(x)$：初始概率分布（噪声）
- $p_1(x)$：目标概率分布（数据）

> 📝 **注解：** 边际概率路径考虑了所有可能的目标数据点。每个数据点 $x_1$ 贡献一个条件概率路径，边际路径是它们的加权平均（权重为数据分布 $p_{\text{data}}$）。

### 3.9 边际向量场

**边际向量场**也是条件向量场的聚合：

$$v_t(x) = \int v_t(x|x_1)\, \frac{p_t(x|x_1)\, p_{\text{data}}(x_1)}{p_t(x)}\,\mathrm{d}x_1$$

权重 $\frac{p_t(x|x_1) p_{\text{data}}(x_1)}{p_t(x)}$ 是后验概率 $p(x_1|x)$——"给定当前位置 $x$，目标数据点 $x_1$ 的概率"。

**直觉：** "根据你现在的位置，你应该往哪走？"

> 📝 **注解：** 边际向量场的直觉非常清晰——它是在当前位置处，对所有可能的目标方向进行加权平均，权重是"到达各目标的可能性"。这就像 GPS 导航：根据你当前的位置，计算最可能的行驶方向。

### 3.10 边际向量场生成边际概率路径

**关键结论：** 边际向量场 $v_t(x)$ **生成**边际概率路径 $p_t(x)$。

即：边际向量场也满足连续性方程。

**推论：** 如果 $p_0(x) = \mathcal{N}(0, I)$（标准高斯噪声）且 $p_1(x) \approx p_{\text{data}}(x)$（数据分布），则边际向量场将噪声传输到数据。

> 📝 **注解：** 这是 Flow Matching 理论的核心——虽然我们无法直接计算边际向量场（因为需要积分），但我们证明了它确实存在且生成正确的概率路径。接下来，我们将利用这个结论推导可处理的损失函数。

### 3.11 推导 Conditional Flow Matching（CFM）

**Flow Matching 损失：**

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t}\left[\|v_\theta(x, t) - v_t(x)\|^2\right]$$

这个损失**不可处理**——因为 $v_t(x)$ 未知。

**关键定理：** FM 损失的梯度等于 CFM 损失的梯度：

$$\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$$

其中 CFM 损失为：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1 \sim p_{\text{data}}, x \sim p_t(\cdot|x_1)}\left[\|v_\theta(x, t) - v_t(x|x_1)\|^2\right]$$

**CFM 损失是可处理的！** 因为条件向量场 $v_t(x|x_1)$ 有解析表达式。

> 📝 **注解推导：** 证明思路与 Lecture 2 中 DSM 的证明类似——展开平方范数，利用边际分布的定义，证明 FM 和 CFM 的梯度相等（差一个与 $\theta$ 无关的常数）。完整推导见 *Flow Matching for Generative Modeling*, Lipman et al., 2022。

### 3.12 Conditional Flow Matching（CFM）

**CFM = Conditional Flow Matching**

**我们现在有了可处理的损失函数！**

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1, \epsilon}\left[\left\|v_\theta\bigl(\mu_t(x_1) + \sigma_t(x_1)\epsilon,\, t\bigr) - v_t\bigl(\mu_t(x_1) + \sigma_t(x_1)\epsilon \mid x_1\bigr)\right\|^2\right]$$

其中 $t \sim \mathcal{U}[0,1]$，$x_1 \sim p_{\text{data}}$，$\epsilon \sim \mathcal{N}(0, I)$。

> 📝 **注解：** CFM 损失完全可计算——我们只需要：
> 1. 从数据集采样 $x_1$
> 2. 从均匀分布采样 $t$
> 3. 从标准正态采样 $\epsilon$
> 4. 构造条件样本 $x_t = \mu_t(x_1) + \sigma_t(x_1)\epsilon$
> 5. 计算神经网络预测 $v_\theta(x_t, t)$ 与条件向量场 $v_t(x_t|x_1)$ 的 MSE

### 3.13 回顾：Flow Matching 的四步策略

与 Lecture 1 的 ELBO 四步法平行：

```
1. 推导简单情况的目标向量场（Dirac → 条件向量场）
2. 用连续性方程构造一般情况的目标向量场
3. 证明 FM 和 CFM 损失的梯度相等 → 可处理！
4. 推导最终损失函数 → 极其简单的 CFM 损失！
```

### 3.14 为什么叫"Flow Matching"？

1. **1:1 映射**：由于向量场的 Lipschitz 连续性，每个初始点对应唯一的轨迹——即"流"
2. **历史原因**：与 Normalizing Flows、Continuous Normalizing Flows 一脉相承

> 📝 **注解：** "Flow"指的是向量场诱导的流映射——从初始点到终点的确定性变换。"Matching"指的是让神经网络的速度场"匹配"目标速度场。整个名字概括了方法的核心：学习一个流，使其匹配目标概率路径。

---

## 四、训练

### 4.1 训练过程

```
1. 采样：
   - 干净图像 x_1 ~ p_data
   - 时间步 t ~ Uniform[0,1]
   - 噪声 ε ~ N(0, I)

2. 构造加噪样本并预测：
   x_t = μ_t(x_1) + σ_t(x_1) · ε
   用 v_θ(x_t, t) 预测条件向量场 v_t(x_t | x_1)

3. 计算损失并反向传播：
   L = ‖v_θ(x_t, t) - v_t(x_t | x_1)‖²
```

> 📝 **注解：** 训练过程与 DDPM/DSM 非常相似——都是采样数据、加噪、预测目标、计算 MSE。关键区别在于预测目标：DDPM 预测噪声 $\epsilon$，DSM 预测 score $s$，Flow Matching 预测速度 $v$。

---

## 五、推理

### 5.1 推理过程

```
1. 采样纯噪声：x_0 ~ N(0, I)

2. 用 Euler 方法数值求解 ODE：
   x_{t+Δt} = x_t + Δt · v_θ(x_t, t)

3. 获得最终图像 x_1
```

> 📝 **注解：** 推理过程就是简单的 ODE 求解。由于 Flow Matching 是确定性的，不需要注入随机噪声（与反向 SDE 不同）。这使得推理更高效，且可以使用更高级的 ODE 求解器。

---

## 六、Rectified Flow

### 6.1 Flow Matching 的问题

**学习复杂度：**
- 交叉路径导致不同的学习重连
- 即使路径不严格交叉也有问题

**推理效率：**
- 路径不是直的 → 需要更多步数来近似
- 没有专门的求解器来拯救

> 📝 **注解：** Flow Matching 的根本问题是：条件向量场可能导致路径交叉——从不同初始点出发的轨迹可能在中间相遇，然后分叉到不同的终点。这增加了学习难度，也使得推理需要更多步数（因为弯曲的路径需要更细的离散化）。

### 6.2 Reflow 过程

> *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, Liu et al., 2022.

**Step 0：** 训练初始模型（"1-rectified flow"）

**Step 1：** 用"1-rectified flow"模型创建配对数据
- 从噪声 $x_0$ 出发，用当前模型生成 $x_1$
- 得到配对 $(x_0, x_1)$

**Step 2：** 在配对数据上训练新模型（"2-rectified flow"）
- 直接学习从 $x_0$ 到 $x_1$ 的直线路径

**Step 3：** 重复 Step 1-2 可选

> 📝 **注解：** Reflow 的直觉：第一次训练的模型可能走弯路，但它的终点是正确的。用这些"正确的配对"重新训练，新模型就可以走直线——因为直线是两点之间最短的路径。每次 Reflow 都让路径更直。

### 6.3 为什么 Reflow 有效？

**性质 1：** 结果映射仍然遵循目标分布

证明思路：
- 定义 $x_1 = \text{Flow}_\theta(x_0)$（模型从 $x_0$ 生成的结果）
- 由唯一性定理（初始条件确定轨迹）和链式法则 + 全期望公式，可证新模型的边际分布不变

**性质 2：** 路径在每次 Reflow 后可证明更直

定义"直度"（straightness）：$\mathbb{E}[\|x_1 - x_0 - v_\theta(x_t, t)\|^2]$

通过望远镜论证（telescoping argument）和方差分解机制，可证直度递减。

> 📝 **注解：** 性质 1 保证了 Reflow 不会"跑偏"——生成的样本仍然来自正确的分布。性质 2 保证了 Reflow 确实让路径更直——这意味着推理时需要更少的步数。

### 6.4 Reflow 的讨论

- 实践中通常做 1-2 次 Reflow，更多次会引入过多误差
- Reflow 后可以使用更简单的推理技术（如 Euler 方法）
- 存在质量/速度的权衡

---

## 七、讨论：范式对比

### 7.1 三种范式对比

| | 离散时间扩散（DDPM） | 基于Score的扩散（SDE） | Flow Matching |
|---|---|---|---|
| **前向过程** | 离散加噪 $z_t = \sqrt{1-\beta_t}z_{t-1} + \sqrt{\beta_t}\epsilon$ | 连续 SDE $\mathrm{d}x = f\,\mathrm{d}t + g\,\mathrm{d}W$ | ODE $\frac{\mathrm{d}x}{\mathrm{d}t} = v_t(x)$ |
| **从干净到噪声** | ✅ | ✅ | ✅ |
| **学习目标** | 噪声 $\epsilon$ | Score $s$ | 速度 $v$ |
| **生成过程** | 逐步去噪 | 反向 SDE / PF-ODE | ODE 求解 |
| **确定性对应** | DDIM | PF-ODE | 已经是确定性的！ |

> 📝 **注解：** Flow Matching 是三种范式中唯一从一开始就是确定性的方法。DDPM 需要DDIM 来获得确定性推理，SDE 需要 PF-ODE 来获得确定性推理，而 Flow Matching 天然就是 ODE——不需要额外转换。

### 7.2 随机插值

> *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*, Albergo et al., 2023.

随机插值是一个统一框架，可以同时涵盖 Flow 和 Diffusion。

> 📝 **注解：** 随机插值提供了 Flow Matching 和扩散模型的统一视角——通过在确定性 ODE 和随机 SDE 之间插值，可以灵活控制生成过程的随机性程度。这是一个活跃的研究方向。

---

## 附录：关键概念速查表

| 概念 | 定义/公式 | 注解 |
|------|---------|------|
| **速度场** | $v_t(x)$：时刻 $t$、位置 $x$ 处的移动方向和速度 | 比 score 信息更丰富 |
| **连续性方程** | $\frac{\partial p_t}{\partial t} + \nabla_x \cdot (p_t v_t) = 0$ | 质量守恒 |
| **条件概率路径** | $p_t(x\|x_1) = \mathcal{N}(\mu_t(x_1), \sigma_t^2(x_1)I)$ | 给定目标数据点的路径 |
| **条件向量场** | $v_t(x\|x_1)$ | 生成条件概率路径的速度场 |
| **边际概率路径** | $p_t(x) = \int p_t(x\|x_1) p_{\text{data}}(x_1)\mathrm{d}x_1$ | 条件路径的聚合 |
| **边际向量场** | $v_t(x) = \int v_t(x\|x_1) p(x_1\|x)\mathrm{d}x_1$ | 条件向量场的加权平均 |
| **CFM 损失** | $\mathbb{E}\[\|v_\theta(x_t, t) - v_t(x_t\|x_1)\|^2\]$ | 可处理的 Flow Matching 损失 |
| **Reflow** | 用当前模型的配对数据重新训练 | 让路径更直 |
| **时间方向** | $t=0$：噪声，$t=1$：数据 | 与扩散模型相反 |
