# [Spring 2026] CME 296 — Lecture 2: Flow Matching（流匹配）

> **来源：** `spring26-cme296-lecture2.pdf`
> **课程：** Stanford CME 296, Spring 2026
> **日期：** 2026-04-10
> **注解版本：** 含详细数学推导与交叉引用

---

## 目录

1. [大背景：Flow Model 是什么？](#一大背景flow-model-是什么)
2. [Flow Matching 矩阵——全局路线图](#二flow-matching-矩阵全局路线图)
3. [概率路径——从噪声到数据的"快照"](#三概率路径从噪声到数据的快照)
4. [向量场——告诉粒子"往哪里走"](#四向量场告诉粒子往哪里走)
5. [连续性方程——概率守恒定律](#五连续性方程概率守恒定律)
6. [损失函数与训练算法——核心中的核心](#六损失函数与训练算法核心中的核心)
7. [采样算法——训练好了怎么用？](#七采样算法训练好了怎么用)
8. [条件 vs 边际——精妙的桥梁](#八条件-vs-边际精妙的桥梁)
9. [核心定理到底在说什么？](#九核心定理到底在说什么)
10. [总结：整个流程的直觉](#十总结整个流程的直觉)
11. [深度注解与补充](#十一深度注解与补充)

---

## 一、大背景：Flow Model 是什么？

### 1.1 本讲主线

Lecture 1 解释了"怎么用 ODE/SDE 定义生成模型"，这一讲解决的是更关键的问题：

> 如果只给我们数据集，而不给真实向量场，怎么把这个模型训练出来？

Flow Matching 的回答是：

1. 先人为设计一条简单、可控的概率路径
2. 为这条路径写出目标向量场
3. 用监督学习去拟合这个目标向量场

它的价值在于把"训练生成模型"变成了一个普通回归问题。

> 📝 **注解：** 这是生成模型训练范式的重大转变。之前的 DDPM 需要设计噪声调度（$\beta_1, \ldots, \beta_T$），Score SDE 需要选择 SDE 形式（VP/VE/VE-SDE），而 Flow Matching 只需要选择一个概率路径——而且最简单的线性路径就足够好。这种简洁性是 Flow Matching 被广泛采用的重要原因。

### 1.2 核心思想——用一条"河流"把噪声变成图片

想象你有一团**随机噪声**（像电视雪花屏），你希望它"流动"一段时间后，变成一张**清晰的图片**。Flow Model 就是设计这样一条"河流"的方法。

**数学描述：**

> **Flow Model（流模型）：**
> $$X_0 \sim p_{\text{init}}, \qquad \mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t$$

**逐项翻译：**

| 符号 | 含义 | 生活比喻 |
|------|------|----------|
| $X_0$ | 起始点（随机噪声） | 一片雪花屏 |
| $p_{\text{init}}$ | 初始分布（通常是高斯分布） | "雪花屏"的统计规律 |
| $u_t^\theta(X_t)$ | **神经网络学到的向量场**（速度场） | 河流在每个位置的"流速和方向" |
| $\theta$ | 神经网络的参数 | 需要通过训练来确定 |
| $\mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t$ | **常微分方程（ODE）** | 一个粒子按照"河流"的速度向前走 |

> **Diffusion Model（扩散模型）** 多了一项随机噪声：
> $$\mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$

其中 $\sigma_t\mathrm{d}W_t$ 是布朗运动带来的随机抖动。Flow Model 是**确定性**的（没有随机抖动），Diffusion Model 是**随机**的。

> 📝 **注解：** 回顾 Lecture 1 的核心关系：$\sigma_t = 0$ 时 Diffusion Model 退化为 Flow Model。本讲主要讨论 Flow Model（ODE 情形），但所有结果都可以自然推广到 SDE 情形。

### 1.3 训练的目标

$$X_0 \sim p_{\text{init}},\quad \mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t \quad \Longrightarrow \quad X_1 \sim p_{\text{data}}$$

**用大白话说：** 找到参数 $\theta$，让粒子从噪声出发（$t=0$），沿着神经网络给出的"河流方向"走到 $t=1$ 时，终点的分布恰好等于真实数据的分布（比如真实图片的分布）。

> 📝 **注解：** 这个目标等价于：让 ODE 推送后的分布 $[\psi_1]_\# p_{\text{init}}$ 等于 $p_{\text{data}}$，其中 $[\psi_1]_\# p_{\text{init}}$ 表示将 $p_{\text{init}}$ 通过流映射 $\psi_1$ 推送后的分布。这是一个分布匹配问题。

### 1.4 为什么直接训练很难？

难点不在于"会不会写 ODE"，而在于：

- 我们并不知道真实边际向量场 $u_t^{\text{target}}(x)$
- 也不知道中间时刻的边际分布 $p_t(x)$
- 真正想要的量都依赖整个数据分布，通常无法直接算

Flow Matching 的策略不是硬算这些边际量，而是：

> 先构造一个容易写出的条件概率路径 $p_t(\cdot \mid z)$，再借条件化把难问题拆掉。

> 📝 **注解：** "边际"（marginal）之所以难算，是因为它需要对所有数据点 $z$ 做积分：$p_t(x) = \int p_t(x|z) p_{\text{data}}(z)\,\mathrm{d}z$。当数据集很大（如 LAION-5B 有 58.5 亿个样本）时，这个积分在计算上是不可行的。Flow Matching 的精妙之处在于：我们不需要计算这个积分，只需要对单个数据点 $z$ 计算条件量，然后通过期望自动得到边际量。

---

## 二、Flow Matching 矩阵——全局路线图

这是整节课最重要的框架，一个 2×3 的表格：

|  | **概率路径** (Probability Path) | **向量场** (Vector Field) | **损失函数** (FM Loss) |
|---|---|---|---|
| **条件的** (Conditional) | $p_t(\cdot\|z)$ | $u_t^{\text{target}}(x\|z)$ | $\mathcal{L}_{\text{CFM}}(\theta)$ |
| **边际的** (Marginal) | $p_t$ | $u_t^{\text{target}}(x)$ | $\mathcal{L}_{\text{FM}}(\theta)$ |

**两个关键术语：**
- **"条件的"（Conditional）** = **针对单个数据点** $z$。就像"如果目标是这张特定的猫图，路径该怎么走？"
- **"边际的"（Marginal）** = **对所有数据点取平均**。就像"综合考虑所有可能的目标图片，路径整体该怎么走？"

> 💡 **核心洞察**：边际的东西很难直接算，但条件的东西很简单！Flow Matching 的精妙之处就是：**通过优化简单的条件损失，隐式地优化了困难的边际损失**。

> 📝 **注解：** 这个 2×3 矩阵是理解 Flow Matching 的关键。每一列（概率路径、向量场、损失函数）都有条件版本和边际版本。条件版本都是"简单可计算"的，边际版本都是"复杂不可计算"的。Flow Matching 的核心定理说的是：**优化条件损失和优化边际损失在参数空间中是等价的**。所以我们只需要关心条件版本！

---

## 三、概率路径——从噪声到数据的"快照"

### 3.1 条件概率路径 $p_t(\cdot|z)$

**直觉：** 固定一个目标数据点 $z$（比如一张猫图），$p_t(\cdot|z)$ 描述的是：在时刻 $t$，粒子"应该在哪里"的概率分布。

- $t=0$ 时：$p_0(\cdot|z) = p_{\text{init}}$ — 一团宽宽的高斯噪声（大雾弥漫）
- $t=1$ 时：$p_1(\cdot|z) \approx \delta(z)$ — 高度集中在数据点 $z$ 附近（雾散了，只看到猫）

**高斯例子（最常用）：**

$$p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\; \beta_t^2 I_d)$$

**逐项解读：**

| 符号 | 含义 | 随时间变化 |
|------|------|-----------|
| $\mathcal{N}(\mu, \Sigma)$ | 高斯（正态）分布 | — |
| $\alpha_t z$ | **均值**：朝目标 $z$ 移动 | $\alpha_0=0$（不靠近），$\alpha_1=1$（完全到达） |
| $\beta_t^2 I_d$ | **方差**：不确定性的大小 | $\beta_0=1$（大雾），$\beta_1 \approx 0$（清晰） |
| $I_d$ | $d$ 维单位矩阵 | 各方向的不确定性相同 |

> 🎯 **一句话理解：** 随着时间从 0 到 1，这个高斯分布的"中心"从原点滑向目标 $z$，同时"宽度"从很大缩到几乎为零——就像一团雾慢慢聚拢到一个点上。

> 📝 **注解：** $\alpha_t$ 和 $\beta_t$ 被称为**调度函数**（schedule functions）。它们必须满足边界条件：
> - $\alpha_0 = 0, \alpha_1 = 1$（$t=0$ 时不靠近目标，$t=1$ 时完全到达）
> - $\beta_0 = 1, \beta_1 = 0$（$t=0$ 时方差最大，$t=1$ 时方差为零）
>
> 此外，$\alpha_t$ 和 $\beta_t$ 需要足够平滑（可微），以保证向量场的存在性。最简单的选择是线性调度 $\alpha_t = t, \beta_t = 1-t$（见 6.3 节）。

### 3.2 边际概率路径 $p_t$

**直觉：** 不再只看一个目标 $z$，而是把所有可能的目标数据点 $z$ 的条件路径"混合"在一起。

$$p_t(x) = \int p_t(x|z)\, p_{\text{data}}(z)\,\mathrm{d}z$$

**用大白话说：**
- 从数据集里每张图 $z$ 都有自己的一团雾（条件路径）
- 把所有这些雾叠加在一起，就是边际路径
- $t=0$：所有雾叠起来 ≈ 一个大高斯（因为每个条件路径在 $t=0$ 都是高斯）
- $t=1$：所有雾都聚拢到各自的目标图 → 叠加结果就是数据分布 $p_{\text{data}}$

> 📝 **注解：** 这就是**全概率公式**（Law of Total Probability）的应用！如果你学过概率论：$p_t(x) = \mathbb{E}_{z \sim p_{\text{data}}}[p_t(x|z)]$。边际路径是条件路径关于数据分布的期望。虽然公式简单，但直接计算这个积分是不可行的——因为 $p_{\text{data}}$ 未知，且数据集太大。Flow Matching 的训练算法通过 Monte Carlo 采样来近似这个期望。

---

## 四、向量场——告诉粒子"往哪里走"

### 4.1 条件向量场 $u_t^{\text{target}}(x|z)$

向量场给出了在每个位置 $x$、每个时刻 $t$ 的"速度方向"。对于高斯概率路径：

$$u_t^{\text{target}}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}x$$

**别被吓到！** 我们来拆解：

| 符号 | 含义 |
|------|------|
| $\dot{\alpha}_t = \frac{\mathrm{d}\alpha_t}{\mathrm{d}t}$ | $\alpha_t$ 对时间 $t$ 的导数（$\alpha_t$ 变化的速率） |
| $\dot{\beta}_t = \frac{\mathrm{d}\beta_t}{\mathrm{d}t}$ | $\beta_t$ 对时间 $t$ 的导数（$\beta_t$ 变化的速率） |
| $z$ | 目标数据点 |
| $x$ | 粒子当前位置 |

**这个公式本质上在说：** 粒子的速度由两部分组成——
1. **朝目标 $z$ 拉** 的分量：$\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z$
2. **跟当前位置 $x$ 相关** 的分量（控制"收缩"或"扩散"）：$\frac{\dot{\beta}_t}{\beta_t}x$

> 📝 **注解：** 这个公式是怎么来的？它来自连续性方程的约束。如果概率路径是 $p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$，那么满足连续性方程 $\frac{\partial p_t}{\partial t} + \text{div}(p_t \cdot u_t) = 0$ 的向量场恰好就是上面这个公式。这是一个可以直接验证的结果——将 $p_t$ 和 $u_t$ 代入连续性方程，两边相等。

### 4.2 为什么这个向量场是对的？（证明思路）

**Step 1：** 验证 ODE 的解（称为"流"）为：

$$\psi_t^{\text{target}}(x_0|z) = \alpha_t z + \beta_t x_0$$

> 意思是：一个起始在 $x_0$ 的粒子，在时刻 $t$ 会走到 $\alpha_t z + \beta_t x_0$。这就是目标 $z$ 和起点 $x_0$ 的**线性混合**！

**Step 2：** 如果 $X_0 = x_0 \sim \mathcal{N}(0, I_d)$，那么：

$$X_t = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z,\; \beta_t^2 I_d) = p_t(\cdot|z) \quad ✅$$

> 📝 **注解：** Step 1 的验证需要计算 $\frac{\mathrm{d}}{\mathrm{d}t}\psi_t$ 并验证它等于 $u_t^{\text{target}}(\psi_t)$：
>
> $$\frac{\mathrm{d}}{\mathrm{d}t}(\alpha_t z + \beta_t x_0) = \dot{\alpha}_t z + \dot{\beta}_t x_0$$
>
> 另一方面，将 $x = \alpha_t z + \beta_t x_0$ 代入向量场公式：
>
> $$u_t^{\text{target}}(\alpha_t z + \beta_t x_0 | z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}(\alpha_t z + \beta_t x_0)$$
> $$= \dot{\alpha}_t z - \frac{\dot{\beta}_t}{\beta_t}\alpha_t z + \frac{\dot{\beta}_t}{\beta_t}\alpha_t z + \dot{\beta}_t x_0 = \dot{\alpha}_t z + \dot{\beta}_t x_0 \quad ✅$$
>
> 两边相等，验证完成！
>
> Step 2 用到的概率知识：如果 $X \sim \mathcal{N}(0, I)$，那么 $aX + b \sim \mathcal{N}(b, a^2 I)$。这是高斯分布的线性变换性质——概率论课上最基础的内容之一！

### 4.3 边际向量场 $u_t^{\text{target}}(x)$

$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z)\,\frac{p_t(x|z)\,p_{\text{data}}(z)}{p_t(x)}\,\mathrm{d}z$$

**解读：** 边际向量场是所有条件向量场的**加权平均**，权重是"在位置 $x$ 处，数据点 $z$ 的后验概率"。

> 直觉：一个粒子在位置 $x$，它不知道自己要去哪个目标 $z$。所以它的速度 = 所有可能目标对应速度的加权平均，权重是"$z$ 是目标"的可能性。

> 📝 **注解：** 权重 $\frac{p_t(x|z)\,p_{\text{data}}(z)}{p_t(x)}$ 就是贝叶斯后验 $p(z|x,t)$——给定粒子在位置 $x$ 和时刻 $t$，目标数据点是 $z$ 的概率。这个公式虽然优美，但**无法直接计算**，因为：
> 1. $p_t(x)$ 需要对所有 $z$ 积分，计算量巨大
> 2. $p_{\text{data}}(z)$ 未知
>
> Flow Matching 的精妙之处就在于：我们**不需要**计算边际向量场！通过核心定理，优化条件损失就等价于优化边际损失。

**⚠️ 重要的是：这个公式无法直接计算！** 因为积分涉及 $p_t(x)$（整个数据分布的混合），太复杂了。但别担心，Flow Matching 的精妙之处就在于绕过了它。

---

## 五、连续性方程——概率守恒定律

$$\frac{\mathrm{d}}{\mathrm{d}t}p_t(x) = -\text{div}(p_t \cdot u_t)(x)$$

**物理直觉（水流比喻）：**

想象概率密度是"水量"，向量场是"水流速度"：
- **左边** $\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)$：位置 $x$ 处的"水量"随时间的变化
- **右边** $-\text{div}(p_t u_t)(x)$：流入减去流出的净水量

> 📝 **注解：** 散度 $\text{div}(p_t u_t) = \nabla \cdot (p_t u_t) = \sum_{i=1}^d \frac{\partial}{\partial x_i}(p_t [u_t]_i)$。散度为正意味着该点是"源"（概率流出），散度为负意味着该点是"汇"（概率流入）。连续性方程保证了概率总量守恒——不会凭空产生或消失。

> 📝 **深度注解：** 连续性方程与 Fokker-Planck 方程的关系：
> - ODE 情形：连续性方程 $\frac{\partial p_t}{\partial t} + \text{div}(p_t u_t) = 0$
> - SDE 情形：Fokker-Planck 方程 $\frac{\partial p_t}{\partial t} = -\text{div}(p_t u_t) + \frac{1}{2}\sigma_t^2 \Delta p_t$
>
> Fokker-Planck 多了一项扩散项 $\frac{1}{2}\sigma_t^2 \Delta p_t$，描述随机噪声导致的概率扩散。当 $\sigma_t = 0$ 时，Fokker-Planck 退化为连续性方程。

这个方程保证了：如果向量场 $u_t$ 和概率路径 $p_t$ 满足连续性方程，那么沿着 $u_t$ 推动粒子，粒子的分布确实会按照 $p_t$ 演化。

> 📝 **注解：** 连续性方程是 Flow Matching 理论的基石。它建立了概率路径 $p_t$ 和向量场 $u_t$ 之间的一一对应关系：给定 $p_t$，可以唯一确定 $u_t$（反之亦然）。这意味着我们只需要设计概率路径，向量场就自动确定了。

---

## 六、损失函数与训练算法——核心中的核心

### 6.1 条件 Flow Matching 损失（通用形式）

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,z,\,x}\Big[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\Big]$$

其中 $t \sim \text{Unif}[0,1]$，$z \sim p_{\text{data}}$，$x \sim p_t(\cdot|z)$。

**翻译成人话：**
- 随机抽一个时间 $t$、一个数据点 $z$、一个采样点 $x$
- 计算神经网络的预测 $u_t^\theta(x)$ 和真实目标 $u_t^{\text{target}}(x|z)$ 之间的**距离的平方**
- 目标：让这个距离尽量小

> 📝 **注解：** 这就是一个**回归问题**！和你在机器学习入门课上学的"最小化均方误差"是一回事。神经网络 $u_t^\theta$ 的输入是 $(x, t)$，输出是一个 $d$ 维向量（速度），目标是 $u_t^{\text{target}}(x|z)$。这种将生成模型训练转化为回归问题的思路，是 Flow Matching 相比之前方法的最大简化。

> 📝 **注解：** 为什么是 $\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2$ 而不是其他损失？平方损失的选择使得最优解 $u_t^{\theta^*}(x) = \mathbb{E}_{z|x,t}[u_t^{\text{target}}(x|z)]$，这恰好是边际向量场 $u_t^{\text{target}}(x)$。这是核心定理成立的关键——只有平方损失才能保证这个等价性。

### 6.2 代入高斯路径

对于高斯条件概率路径，采样 $x \sim p_t(\cdot|z)$ 等价于：

$$x = \alpha_t z + \beta_t \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I_d)$$

> 📝 **注解：** 这叫**重参数化技巧**（Reparameterization Trick）：与其从复杂分布 $\mathcal{N}(\alpha_t z, \beta_t^2 I)$ 采样，不如先采一个标准噪声 $\epsilon \sim \mathcal{N}(0, I)$，再做线性变换 $x = \alpha_t z + \beta_t \epsilon$。这个技巧的好处是：1) 采样更简单；2) 可以对 $\epsilon$ 求梯度，便于反向传播。重参数化技巧也是 VAE 训练的关键，详见 [VAE_Tutorial.md](notes/VAE_Tutorial.md)。

代入后，目标向量场简化为：

$$u_t^{\text{target}}(x|z) = \dot{\alpha}_t z + \dot{\beta}_t \epsilon$$

损失变成：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I_d)}\Big[\|u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon)\|^2\Big]$$

**解读：**
- 神经网络的输入：$\alpha_t z + \beta_t \epsilon$（噪声和数据的混合物）
- 神经网络应该输出：$\dot{\alpha}_t z + \dot{\beta}_t \epsilon$（对应的"速度"）

> 📝 **注解：** 注意目标 $\dot{\alpha}_t z + \dot{\beta}_t \epsilon$ 中，$z$ 和 $\epsilon$ 都是已知的（从数据集采样 $z$，从标准正态采样 $\epsilon$），所以目标是**完全可计算的**。这正是 Flow Matching 的优势——不需要任何难以计算的边际量。

### 6.3 ⭐ 直线调度 (Straight Line Schedule)——最简单也最常用

令 $\alpha_t = t$，$\beta_t = 1-t$（最简单的线性插值），此时：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,z,\,\epsilon}\Big[\|u_t^\theta\big(t z + (1-t)\epsilon\big) - (z - \epsilon)\|^2\Big]$$

**这是整节课最关键的公式！来详细拆解：**

| 公式部分 | 含义 | 直觉 |
|---------|------|------|
| $tz + (1-t)\epsilon$ | 噪声和数据的**线性插值** | $t=0$ 时全是噪声，$t=1$ 时全是数据 |
| $z - \epsilon$ | 数据减去噪声 = **速度目标** | 粒子应该以"数据减噪声"的方向走 |
| $u_t^\theta(\cdot)$ | 神经网络预测的速度 | 网络要学会预测这个方向 |

> 🎯 **一句话总结：** 给神经网络看一个"半成品"（噪声和图片的混合），让它预测"从噪声到图片的方向"。如此而已！

> 📝 **注解：** 为什么叫"直线调度"？因为流映射 $\psi_t(x_0|z) = tz + (1-t)x_0$ 是从 $x_0$ 到 $z$ 的直线插值。粒子沿着直线从噪声走向数据！这是最简单的路径，但实验表明它已经足够好。更复杂的路径（如最优传输路径）可能进一步提升效率，但直线调度是默认选择。

> 📝 **注解：** 直线调度下，目标速度 $z - \epsilon$ **与时间 $t$ 无关**！这意味着神经网络在不同时刻应该预测同一个方向。这是一个非常强的先验——它简化了学习问题，因为网络不需要学习"什么时候该往哪里走"，只需要学习"该往哪里走"。

### 6.4 完整训练算法（Algorithm 4）

```
输入: 数据集 z ~ p_data, 神经网络 u_t^θ
对每个 mini-batch:
    1. 采样一个数据点 z（从数据集中取一张图片）
    2. 采样一个随机时间 t ~ Uniform[0,1]
    3. 采样一个噪声 ε ~ N(0, I_d)
    4. 构造混合样本 x = tz + (1-t)ε
    5. 计算损失 L(θ) = ‖u_t^θ(x) - (z - ε)‖²
    6. 用梯度下降更新参数 θ
```

> 就这么简单！**没有复杂的积分，没有难以计算的边际分布。** 每一步都是采样 → 前向传播 → 算距离 → 反向传播。和训练一个普通神经网络的流程完全一样。

> 📝 **注解：** 与 DDPM 训练的对比：
>
> | | DDPM | Flow Matching |
> |---|---|---|
> | 需要设计噪声调度 | 是（$\beta_1, \ldots, \beta_T$） | 否（$\alpha_t = t, \beta_t = 1-t$） |
> | 损失函数 | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | $\|u_t^\theta(x) - (z - \epsilon)\|^2$ |
> | 目标 | 预测噪声 $\epsilon$ | 预测速度 $z - \epsilon$ |
> | 时间离散性 | 离散 $t = 1, \ldots, T$ | 连续 $t \in [0, 1]$ |
> | 超参数数量 | 多（$T$ 个 $\beta$ 值） | 少（只有 $\alpha_t, \beta_t$ 的选择） |
>
> 详见 [DDPM_Detailed.md](notes/DDPM_Detailed.md) 和 [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md)。

---

## 七、采样算法——训练好了怎么用？

训练完成后，用 **Euler 方法** 模拟 ODE 来生成新样本：

```
输入: 神经网络向量场 u_t^θ, 步数 n
1. 设 t = 0, 步长 h = 1/n
2. 从 p_init 采样 X_0（一团随机噪声）
3. 循环 n 步:
      X_{t+h} = X_t + h · u_t^θ(X_t)    ← 用神经网络预测速度，走一小步
      t ← t + h
4. 返回 X_1（最终结果就是生成的图片！）
```

> 📝 **注解：** 采样算法与 Lecture 1 中的 Flow Model 采样算法完全一致。训练阶段学到了向量场 $u_t^\theta$，采样阶段就是用 Euler 方法模拟这个 ODE。步数 $n$ 的选择取决于精度和速度的权衡：
> - $n = 1$：一步生成（最快，但质量差）
> - $n = 20$：20 步生成（质量已经很好）
> - $n = 1000$：1000 步生成（质量最好，但很慢）
>
> 实践中，20-50 步通常是质量与速度的最佳平衡点。

> 📝 **注解：** 与 DDPM 采样的对比：DDPM 通常需要 1000 步才能生成高质量图片，而 Flow Matching 只需 20-50 步。这是因为 Flow Matching 使用 ODE 采样（确定性），而 DDPM 使用 SDE 采样（随机性），ODE 的数值收敛速度比 SDE 快得多。DDIM 通过去掉随机性将 DDPM 的采样步数从 1000 降到 20-50，本质上就是将 SDE 采样改为了 ODE 采样。详见 [DDIM_Detailed.md](notes/DDIM_Detailed.md)。

---

## 八、条件 vs 边际——精妙的桥梁

最后的大图景：

|  | 条件的 (Conditional) | 边际的 (Marginal) |
|---|---|---|
| **概率路径** | $p_t(\cdot\|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$ ✅ 简单 | $p_t(x) = \int p_t(x\|z)p_{\text{data}}(z)\mathrm{d}z$ ❌ 难算 |
| **向量场** | $u_t^{\text{target}}(x\|z)$ = 解析公式 ✅ 简单 | $u_t^{\text{target}}(x)$ = 复杂积分 ❌ 难算 |
| **损失函数** | $\mathcal{L}_{\text{CFM}}$ ✅ 可以直接计算 | $\mathcal{L}_{\text{FM}}$ ❌ 无法直接计算 |
| **关系** | **优化条件损失...** | **...等价于隐式优化边际损失！** |

> 🏆 **Flow Matching 的核心定理：** 最小化条件 FM 损失 $\mathcal{L}_{\text{CFM}}$ 和最小化边际 FM 损失 $\mathcal{L}_{\text{FM}}$ 对参数 $\theta$ 来说是**等价的**。我们只需要优化简单的条件损失，就自动学到了整体的边际向量场！

> 📝 **注解：** 核心定理的证明思路：
>
> 1. 定义边际 FM 损失：$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t}\Big[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2\Big]$
>
> 2. 展开平方项：$\|u_t^\theta - u_t^{\text{target}}(x)\|^2 = \|u_t^\theta\|^2 - 2\langle u_t^\theta, u_t^{\text{target}}(x)\rangle + \|u_t^{\text{target}}(x)\|^2$
>
> 3. 关键观察：$\mathbb{E}_{x \sim p_t}[u_t^{\text{target}}(x)]$ 的交叉项可以通过条件化重写为 $\mathbb{E}_{z, x \sim p_t(\cdot|z)}[u_t^{\text{target}}(x|z)]$
>
> 4. 因此 $\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta) + C$，其中 $C$ 是与 $\theta$ 无关的常数
>
> 5. 所以两个损失关于 $\theta$ 的梯度方向相同，最小化其中一个等价于最小化另一个
>
> 详见 [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md) 中的完整证明。

---

## 九、核心定理到底在说什么？

这一定理最容易被误解成"条件模型等于边际模型"，其实不是。

更准确地说，它表达的是：

- 训练时，我们喂给网络的是带终点标签 $z$ 的条件监督
- 推断时，网络输出的是不依赖具体 $z$ 的统一向量场
- 在平方损失下，这种条件监督的最优解恰好对应边际向量场

所以 Flow Matching 的真正贡献不是某个特殊公式，而是一种训练原则：

> **用容易采样、容易求解析式的条件问题，替代原本难以直接优化的边际问题。**

> 📝 **注解：** 这个原则可以类比于 EM 算法（Expectation-Maximization）的思想：直接优化边际似然很难，但通过引入隐变量（这里是目标数据点 $z$），将问题转化为条件期望的优化，就变得可行了。Flow Matching 中的 $z$ 就扮演了 EM 算法中隐变量的角色。

> 📝 **注解：** 另一个类比是监督学习中的"教师强制"（teacher forcing）：训练时给网络看正确答案（条件信息 $z$），推断时网络不需要答案就能做出正确预测。Flow Matching 的训练也是类似的——训练时利用 $z$ 的信息构造监督信号，推断时网络已经学会了不依赖 $z$ 的边际向量场。

---

## 十、总结：整个流程的直觉

```
          训练阶段                          生成阶段
    ┌─────────────────┐            ┌─────────────────┐
    │ 噪声 ε          │            │ 随机噪声 X₀      │
    │   +             │            │     │            │
    │ 数据 z          │            │     ▼ 沿ODE走    │
    │   ↓             │            │   X₀ + h·u(X₀)  │
    │ 混合: tz+(1-t)ε │            │     │            │
    │   ↓             │            │     ▼ ...        │
    │ 网络预测速度     │            │     │            │
    │   ↓             │            │     ▼            │
    │ 目标: z - ε     │            │   X₁ = 生成图片！ │
    │   ↓             │            └─────────────────┘
    │ 最小化差距       │
    └─────────────────┘
```

**实际应用：** Meta 的 MovieGen（视频生成）和 Stability AI 的 Stable Diffusion 3（图像生成）都是用这个算法训练的！

> 📝 **注解：** Stable Diffusion 3 使用了 Rectified Flow（直线流），这是 Flow Matching 的一个变体，本质上就是直线调度 $\alpha_t = t, \beta_t = 1-t$ 的 Flow Matching。Rectified Flow 还提出了"reflow"技术——用已训练模型的采样结果重新训练，使路径更直，从而进一步减少采样步数。

---

## 十一、深度注解与补充

### 11.1 Flow Matching 与 Optimal Transport 的联系

直线调度 $\alpha_t = t, \beta_t = 1-t$ 对应的流映射 $\psi_t(x_0) = tz + (1-t)x_0$ 是从 $x_0$ 到 $z$ 的直线。但这条直线不一定是最优传输（Optimal Transport, OT）路径。

> 📝 **注解：** 最优传输理论寻找的是将一个分布"搬运"到另一个分布的"最省力"方式。在 2-Wasserstein 距离下，OT 映射使得传输代价 $\int \|x - T(x)\|^2 p_0(x)\,\mathrm{d}x$ 最小。直线调度在条件意义下是 OT 的（每个条件路径都是直线），但在边际意义下不一定是 OT 的（因为不同条件路径可能交叉）。Reflow 技术可以减少路径交叉，使边际路径更接近 OT。

### 11.2 不同调度函数的选择

除了直线调度，还有其他选择：

| 调度 | $\alpha_t$ | $\beta_t$ | 特点 |
|------|-----------|-----------|------|
| **直线（OT）** | $t$ | $1-t$ | 最简单，最常用 |
| **余弦调度** | $\frac{1}{2}(1 - \cos\pi t)$ | $\sqrt{1 - \alpha_t^2}$ | 更平滑的过渡 |
| **VP 调度** | $e^{-\frac{1}{2}T(t)}$ | $\sqrt{1 - e^{-T(t)}}$ | 对应 DDPM 的 VP-SDE |
| **可学习调度** | 神经网络参数化 | 神经网络参数化 | 最灵活，但更复杂 |

> 📝 **注解：** VP 调度将 Flow Matching 与 DDPM/Score SDE 联系起来。当选择 VP 调度时，Flow Matching 的条件向量场等价于 DDPM 的去噪目标。这意味着 DDPM 可以看作 Flow Matching 的一个特例——选择了特定概率路径的 Flow Matching。详见 [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md) 中的详细对比。

### 11.3 条件生成如何加入？

本讲聚焦无条件生成，但实际应用中条件生成更为常见。在 Flow Matching 框架中加入条件非常简单：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,z,\,\epsilon,\,y}\Big[\|u_t^\theta(x, y) - u_t^{\text{target}}(x|z)\|^2\Big]$$

其中 $y$ 是条件变量（如文本 prompt），$u_t^\theta(x, y)$ 是以 $y$ 为条件的神经网络。

> 📝 **注解：** 在实践中，条件 $y$ 通常通过交叉注意力（cross-attention）机制注入神经网络。Stable Diffusion 的架构就是：U-Net（主干）+ CLIP 文本编码器（条件）+ 交叉注意力（注入方式）。条件生成不影响 Flow Matching 的核心定理——定理在条件版本下同样成立。

### 11.4 Flow Matching 与 DDPM/Score SDE 的统一视角

```
                    Flow Matching（统一框架）
                         │
            ┌────────────┼────────────┐
            │            │            │
     直线调度        VP 调度       VE 调度
     (Rectified Flow)  (等价于DDPM)  (等价于Score SDE-VE)
            │            │            │
     确定性ODE       随机SDE       随机SDE
     少步采样        多步采样       多步采样
```

> 📝 **注解：** Flow Matching 不是一个全新的方法，而是一个**统一框架**。通过选择不同的概率路径（调度函数），Flow Matching 可以恢复出 DDPM、Score SDE、DDIM 等已有方法。这种统一视角有助于理解不同方法之间的联系，也为设计新方法提供了清晰的指导。详见 [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md)。

### 11.5 实践中的技巧

| 技巧 | 描述 | 作用 |
|------|------|------|
| **时间采样加权** | 不用均匀 $t \sim \text{Unif}[0,1]$，而是用 $\logit$ 正态采样 | 更关注中间时刻（变化最快的阶段） |
| **Reflow** | 用已训练模型生成 $(x_0, x_1)$ 对，重新训练 | 使路径更直，减少采样步数 |
| **自适应步长** | 用 Runge-Kutta 方法代替 Euler | 更精确的 ODE 求解 |
| **分类器自由引导** | 同时训练条件和无条件模型，推理时线性外推 | 提高条件生成的质量和可控性 |

> 📝 **注解：** 时间采样加权是实践中最重要的技巧之一。直觉上，$t$ 接近 0 或 1 时，混合样本 $tz + (1-t)\epsilon$ 要么几乎是纯噪声，要么几乎是纯数据，学习信号较弱。中间时刻的混合样本包含最多的信息，因此应该被更频繁地采样。$\logit$ 正态采样 $t = \text{sigmoid}(\mu + \sigma \cdot \epsilon)$（$\epsilon \sim \mathcal{N}(0,1)$）可以实现这一点。

### 11.6 复习时先抓住三件事

1. **概率路径** $p_t$ 决定了"中间态长什么样"——从噪声到数据的渐进过渡。
2. **向量场** $u_t$ 决定了"粒子下一步往哪里走"——由概率路径通过连续性方程唯一确定。
3. **Flow Matching 的训练诀窍**是：不直接学难算的边际目标，而是通过条件路径做可计算监督——优化条件损失等价于优化边际损失。

---

> **相关阅读：**
> - [Lecture 1 注解文档](spring26-cme296-lecture1-annotated.md) — Flow/Diffusion 模型的数学基础
> - [DDPM_Detailed.md](notes/DDPM_Detailed.md) — 第一代扩散模型的完整推导
> - [DDIM_Detailed.md](notes/DDIM_Detailed.md) — 从 DDPM 到 ODE 的桥梁
> - [ScoreSDE_Detailed.md](notes/ScoreSDE_Detailed.md) — SDE 统一框架
> - [FlowMatching_Detailed.md](notes/FlowMatching_Detailed.md) — Flow Matching 的完整数学推导
> - [VAE_Tutorial.md](notes/VAE_Tutorial.md) — 变分自编码器与重参数化技巧
