# 第四代：Flow Matching (Conditional Flow Matching)

> **论文：** Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
>
> **地位：** 近年非常重要的生成模型训练范式之一，在部分前沿模型与公开实现中被采用

---

## 目录

1. [核心思想](#一核心思想)
2. [数学框架](#二数学框架)
3. [条件向量场的推导](#三条件向量场的推导)
4. [训练方法](#四训练方法)
5. [CFM ≡ FM 等价性证明](#五cfm--fm-等价性证明)
6. [不同概率路径的选择](#六不同概率路径的选择)
7. [与前几代的详细对比](#七与前几代的详细对比)
8. [为什么 Flow Matching 更好](#八为什么-flow-matching-更好)
9. [Flow Matching 的完整变量表](#九flow-matching-的完整变量表)
10. [Flow Matching 的优缺点总结](#十flow-matching-的优缺点总结)
11. [实践代码示例](#十一实践代码示例)
12. [四代演进全景总结](#十二四代演进全景总结)

---

## 一、核心思想

### 1.1 从"预测噪声"到"回归向量场"

```
前三代:                              Flow Matching:
┌──────────────────────┐            ┌──────────────────────┐
│ 学什么?              │            │ 学什么?              │
│ → 预测噪声 ε         │   范式转换   │ → 回归向量场 u        │
│                      │    ──→      │                      │
│ 损失函数:             │            │ 损失函数:             │
│ L = ‖ε - εθ(x,t)‖²  │            │ L = ‖u - u^{target}‖²│
│                      │            │                      │
│ 需要设计:             │            │ 需要设计:             │
│ β_t 或 g(t)          │            │ 无！                 │
│ (噪声调度)           │            │ (自动确定)           │
└──────────────────────┘            └──────────────────────┘
 复杂的超参数选择                    简洁的回归问题
```

### 1.2 核心直觉

**Flow Matching 的训练过程就像教一个学生走迷宫：**

```
目标数据 z（如一张猫图）     噪声 x₀（起点）
    ● ─────────────────────────── ●

    ↑ 我们告诉学生："从 x 出发，朝 z 的方向走"

    学生学到的就是向量场 u_t(x)

    训练时：随机选一个中间点 x = t·z + (1-t)·x₀
            让学生预测方向 u_t(x)
            真实方向是 u^{target} = z - x₀
            损失 = ‖预测方向 - 真实方向‖²
```

### 1.3 连续正则化流（CNF）背景

Flow Matching 是**连续正则化流（Continuous Normalizing Flow, CNF）**的新训练范式。

**CNF 的核心思想：** 用一个 ODE 将简单分布（如高斯）变换为复杂分布（如数据分布）：

$$\frac{\mathrm{d}x}{\mathrm{d}t} = v_t(x), \quad x(0) \sim p_{\text{simple}}, \quad x(1) \sim p_{\text{data}}$$

**之前的 CNF 训练方法（FFJORD, Grathwohl et al. 2018）：**
- 需要计算 ODE 的雅可比行列式（用 Hutchinson 迹估计器）
- 训练不稳定，计算昂贵
- 需要模拟 ODE 来计算损失

**Flow Matching 的突破：**
- **不需要计算雅可比行列式**
- **不需要模拟 ODE 来训练**
- 只需要回归向量场，训练极其简单

---

## 二、数学框架

### 2.1 条件概率路径

**定义：** 一个随时间 $t \in [0,1]$ 变化的概率分布族 $p_t(x|z)$，满足：

| 时间 | 分布 | 含义 |
|------|------|------|
| $t=0$ | $p_0(x|z) = p_{\text{simple}}(x)$ | 从简单分布出发（如高斯） |
| $t=1$ | $p_1(x|z) \approx \delta_z(x)$ | 收敛到目标数据点 $z$ |
| 中间 | 平滑插值 | 连续过渡 |

### 2.2 两种概率路径选择

Flow Matching 的灵活性在于可以选择不同的概率路径。以下是两种最重要的选择：

#### 选择 1：OT-CFM（最优传输条件流匹配）

**概率路径：**

$$p_t(x|z) = \mathcal{N}(x; tz, (1-t)^2 \sigma^2 I)$$

当 $\sigma=1$ 时：

$$p_t(x|z) = \mathcal{N}(x; tz, (1-t)^2 I)$$

**特点：**
- 均值从 0 线性移动到 $z$
- 方差从 1 线性收缩到 0
- 对应"最优传输"路径（两点之间的直线）
- $t=0$ 时：$p_0 = \mathcal{N}(0, I)$
- $t=1$ 时：$p_1 \approx \delta_z$（方差趋近 0）

#### 选择 2：FM（常方差流匹配，Lipman et al. 2022 原始版本）

**概率路径：**

$$p_t(x|z) = \mathcal{N}(x; tz, \sigma^2 I)$$

其中 $\sigma$ 是一个固定常数。

**特点：**
- 均值从 0 线性移动到 $z$
- 方差保持常数 $\sigma^2$
- $t=0$ 时：$p_0 = \mathcal{N}(0, \sigma^2 I)$（不是标准高斯！）
- $t=1$ 时：$p_1 = \mathcal{N}(z, \sigma^2 I)$（不是 delta 分布！）

**两种路径的对比：**

| 性质 | OT-CFM | FM（常方差） |
|------|--------|-------------|
| 方差 | $(1-t)^2$（时变） | $\sigma^2$（常数） |
| $t=0$ 分布 | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma^2 I)$ |
| $t=1$ 分布 | $\approx \delta_z$ | $\mathcal{N}(z, \sigma^2 I)$ |
| 向量场 | $u_t = \frac{z-x}{1-t}$（时变） | $u_t = z$（常数！） |
| 实践效果 | **更好**（路径更直） | 理论优美但效果略差 |

> ⚠️ **本文后续主要使用 OT-CFM**，因为它在很多公开实验里更稳、更直观。实际工业实现是否采用、采用到什么程度，要看具体模型和版本。

---

## 三、条件向量场的推导

### 3.1 OT-CFM 的条件向量场

**这是 Flow Matching 最优雅的地方——向量场可以直接算出来，不需要学习！**

**推导：** 考虑 ODE：

$$\frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t|z), \quad X_0 \sim p_{\text{simple}}$$

如果 $X_t$ 的分布恰好是 $p_t(\cdot|z) = \mathcal{N}(tz, (1-t)^2 I)$，那么 $X_t$ 可以表示为：

$$X_t = t \cdot z + (1-t) \cdot X_0, \quad X_0 \sim \mathcal{N}(0, I)$$

**验证分布：**

- 均值：$\mathbb{E}[X_t] = tz + (1-t) \cdot 0 = tz$ ✓
- 方差：$\text{Var}(X_t) = (1-t)^2 \cdot I$ ✓

**对时间求导：**

$$\frac{\mathrm{d}X_t}{\mathrm{d}t} = z - X_0$$

**用 $(X_t, t)$ 表示：** 由 $X_t = tz + (1-t)X_0$ 得到 $X_0 = \frac{X_t - tz}{1-t}$，代入：

$$\frac{\mathrm{d}X_t}{\mathrm{d}t} = z - \frac{X_t - tz}{1-t} = \frac{z(1-t) - X_t + tz}{1-t} = \frac{z - X_t}{1-t}$$

$$\boxed{u_t^{\text{target}}(x|z) = \frac{z - x}{1-t}}$$

**验证：**

| 性质 | 验证 |
|------|------|
| $t=0$ 时 | $u_0(x \mid z) = z - x$（把粒子推向 $z$） |
| $t \to 1$ 时 | $u_t \to \infty$（需要数值处理，见后文） |
| 方向正确 | 当 $x$ 在 $z$ 左边时推力向右，反之亦然 |
| 大小合理 | 离 $z$ 越远，推力越大 |

### 3.2 FM（常方差）的条件向量场

对于 $p_t(x|z) = \mathcal{N}(tz, \sigma^2 I)$，$X_t = tz + \sigma \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

$$\frac{\mathrm{d}X_t}{\mathrm{d}t} = z$$

$$\boxed{u_t^{\text{target}}(x|z) = z}$$

**这是一个常数向量场！** 无论当前位置 $x$ 在哪里，方向始终指向 $z$。

### 3.3 边际向量场与边际分布

**边际分布：**

$$p_t(x) = \int p_t(x|z)p_{\text{data}}(z)\,\mathrm{d}z$$

**边际向量场：**

$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z)\,\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\,\mathrm{d}z$$

> ⚠️ **边际向量场无法直接计算**（积分涉及未知的 $p_t(x)$）。但这正是 Flow Matching 的精妙之处——我们不需要它！

---

## 四、训练方法

### 4.1 条件 Flow Matching 损失

**核心思想：** 不直接学习边际向量场 $u_t(x)$，而是学习条件向量场 $u_t(x|z)$。因为条件向量场有解析解，可以直接作为训练目标。

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,z,x}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\right]$$

其中采样过程为：
1. 采样 $z \sim p_{\text{data}}$
2. 采样 $t \sim \text{Uniform}[0,1]$
3. 采样 $x_0 \sim p_{\text{simple}}$（对于 OT-CFM：$x_0 = \epsilon \sim \mathcal{N}(0, I)$）
4. 计算 $x = tz + (1-t)x_0$（OT-CFM 的插值）

### 4.2 OT-CFM 的简化形式

对于 OT-CFM，$x = tz + (1-t)\epsilon$，目标向量场 $u_t^{\text{target}} = \frac{z - x}{1-t}$。

代入 $x = tz + (1-t)\epsilon$：

$$u_t^{\text{target}} = \frac{z - (tz + (1-t)\epsilon)}{1-t} = \frac{z - tz - (1-t)\epsilon}{1-t} = \frac{(1-t)(z - \epsilon)}{1-t} = z - \epsilon$$

**最终损失函数（极其简洁）：**

$$\boxed{\mathcal{L}_{\text{OT-CFM}}(\theta) = \mathbb{E}_{t,z,\epsilon}\left[\left\|u_t^\theta(tz + (1-t)\epsilon, t) - (z - \epsilon)\right\|^2\right]}$$

| 公式部分 | 含义 |
|----------|------|
| $tz + (1-t)\epsilon$ | **输入**：噪声和数据的线性插值 |
| $u_t^\theta(\cdot, t)$ | **神经网络输出**：预测的向量场 |
| $z - \epsilon$ | **目标**：真实方向（数据减去噪声） |

**注意：** 目标 $z - \epsilon$ **不依赖时间 $t$**！这意味着神经网络在不同时间步需要预测相同的方向，只是输入的 $x$ 在不同时间步不同。

### 4.3 完整训练算法

**算法：Conditional Flow Matching 训练**

| 步骤 | 操作 |
|------|------|
| **输入** | 数据集 $\{z_i\}$，网络 $u_t^\theta$，迭代次数 $M$ |
| **循环** | 对每个 epoch / mini-batch： |
| ① | 采样 $z \sim p_{\text{data}}$（从数据集中取一个样本） |
| ② | 采样 $t \sim \text{Uniform}[0,1]$（均匀采一个时间） |
| ③ | 采样 $\epsilon \sim \mathcal{N}(0, I_d)$（标准高斯噪声） |
| ④ | 构造混合样本 $x = tz + (1-t)\epsilon$ |
| ⑤ | 计算损失 $\mathcal{L} = \Vert u_t^\theta(x, t) - (z - \epsilon)\Vert^2$ |
| ⑥ | 反向传播更新参数 $\theta$ |

> 🎯 **这就是全部了！没有 $\beta_t$、没有 $\bar{\alpha}_t$、没有复杂的辅助变量。就是一个普通的回归问题。**

### 4.4 t→1 时的数值稳定性

当 $t \to 1$ 时，$u_t = (z-x)/(1-t)$ 的分母趋近 0。但在实践中这不是问题，因为：

1. **训练时**：目标 $z - \epsilon$ 不依赖 $t$，所以不存在数值问题
2. **采样时**：使用 $x_{t+h} = x_t + h \cdot u_t^\theta(x_t, t)$，当 $t$ 接近 1 时停止积分
3. **实践技巧**：将 $t$ 的采样范围限制在 $[0, 1-\epsilon]$（如 $\epsilon = 10^{-4}$）

---

## 五、CFM ≡ FM 等价性证明

这是 Flow Matching 的核心定理：**最小化条件损失等价于最小化边际损失。**

### 5.1 两种损失的定义

**边际损失（Marginal Flow Matching）：**

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t(x)}\left[\|u_t^\theta(x, t) - u_t^{\text{target}}(x)\|^2\right]$$

**条件损失（Conditional Flow Matching）：**

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z \sim p_{\text{data}}, x \sim p_t(x|z)}\left[\|u_t^\theta(x, t) - u_t^{\text{target}}(x|z)\|^2\right]$$

### 5.2 定理陈述

**定理（Lipman et al., 2022, Theorem 1）：**

$$\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta)$$

即两个损失的梯度相同，因此最小化任一个都会收敛到相同的解。

### 5.3 证明思路

**Step 1：** 展开边际损失：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_t\left[\int p_t(x) \|u_t^\theta - u_t\|^2 \mathrm{d}x\right]$$

**Step 2：** 展开条件损失：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_t\left[\int \int p_t(x|z) p_{\text{data}}(z) \|u_t^\theta - u_t(\cdot|z)\|^2 \mathrm{d}x \, \mathrm{d}z\right]$$

**Step 3：** 对 $\mathcal{L}_{\text{CFM}}$ 展开 $\|u_t^\theta - u_t(\cdot|z)\|^2$：

$$= \mathbb{E}_t\left[\int p_t(x) \|u_t^\theta\|^2 \mathrm{d}x - 2\int p_t(x) u_t^\theta \cdot u_t(x) \mathrm{d}x + C_1\right]$$

**Step 4：** 对 $\mathcal{L}_{\text{FM}}$ 展开 $\|u_t^\theta - u_t\|^2$：

$$= \mathbb{E}_t\left[\int p_t(x) \|u_t^\theta\|^2 \mathrm{d}x - 2\int p_t(x) u_t^\theta \cdot u_t(x) \mathrm{d}x + C_2\right]$$

**Step 5：** 关键等式（利用边际向量场的定义）：

$$u_t(x) = \int u_t(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \mathrm{d}z$$

因此两个损失中关于 $u_t^\theta$ 的线性项相同，而常数项 $C_1, C_2$ 不影响梯度。

**结论：** $\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$。

### 5.4 证明的意义

1. **我们不需要计算边际向量场 $u_t(x)$**——只需用条件向量场 $u_t(x|z)$ 作为目标
2. **条件向量场有解析解**——不需要任何近似
3. **训练极其简单**——就是普通的回归问题

---

## 六、不同概率路径的选择

### 6.1 路径选择总览

| 路径 | 条件概率 $p_t(x \mid z)$ | 条件向量场 $u_t(x \mid z)$ | 特点 |
|------|----------------------|------------------------|------|
| **OT-CFM** | $\mathcal{N}(tz, (1-t)^2 I)$ | $\frac{z-x}{1-t}$ | 路径最直，效果最好 |
| **FM** | $\mathcal{N}(tz, \sigma^2 I)$ | $z$ | 常数向量场，理论优美 |
| **VP-CFM** | 对应 VP-SDE 的路径 | 对应 VP-SDE 的向量场 | 与 DDPM 等价 |
| **自定义** | 任意高斯路径 | 由路径决定 | 灵活但需推导 |

### 6.2 VP-CFM：与 DDPM 的等价关系

VP-CFM 选择与 VP-SDE 相同的概率路径：

$$p_t(x|z) = \mathcal{N}(x; \alpha_t z, \sigma_t^2 I)$$

其中 $\alpha_t = e^{-\frac{1}{2}\int_0^t \beta(s)ds}$，$\sigma_t^2 = 1 - \alpha_t^2$。

对应的条件向量场为：

$$u_t(x|z) = \frac{\dot{\alpha}_t}{\alpha_t}(x - z \cdot \frac{\dot{\alpha}_t \sigma_t^2}{\alpha_t(1-\alpha_t^2)}(x - \alpha_t z))$$

**当使用 VP-CFM 训练时，结果与 DDPM 完全等价！** 这说明 Flow Matching 是 DDPM 的超集。

### 6.3 Mini-batch OT

当源分布和目标分布都是经验分布（有限样本）时，可以用**最优传输（Optimal Transport）**匹配代替随机配对：

1. 从数据集中采样一个 mini-batch $\{z_1, \ldots, z_B\}$
2. 从噪声中采样 $\{\epsilon_1, \ldots, \epsilon_B\}$
3. 计算 $\{z_i\}$ 和 $\{\epsilon_i\}$ 之间的最优传输匹配
4. 用匹配后的配对训练

**优势：** 减少路径交叉，使训练更高效，生成质量更高。

---

## 七、与前几代的详细对比

### 7.1 公式对比总表

| 项目 | DDPM | DDIM | Score SDE | **Flow Matching** |
|------|------|------|-----------|-------------------|
| **年份** | 2020 | 2021 | 2021 | **2022+** |
| **数学框架** | 离散马尔可夫链 | 离散非马尔可夫 | 连续 SDE/ODE | **连续 ODE** |
| **前向过程** | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | 同左 | $\mathrm{d}x = f\,\mathrm{d}t + g\,\mathrm{d}W_t$ | $p_t(x \mid z) = \mathcal{N}(tz, (1-t)^2 I)$ |
| **学习目标** | 噪声 $\epsilon_\theta$ | 噪声 $\epsilon_\theta$ | Score $s_\theta$ | **向量场 $u_t^\theta$** |
| **损失函数** | $\Vert\epsilon - \epsilon_\theta\Vert^2$ | 同左 | $\lambda\Vert s_\theta - \nabla\log p\Vert^2$ | **$\Vert u_t^\theta - u^{\text{target}}\Vert^2$** |
| **噪声调度** | ⚠️ 需设计 $\beta_t$ | 复用 DDPM | ⚠️ 需设计 $g(t)$ | **❌ 无需设计！** |
| **辅助变量** | $\alpha_t, \bar{\alpha}_t, \tilde{\beta}_t...$ | 同左 + $\eta$ | $f, g, \lambda$ | **无！** |
| **采样步数** | 1000 | 50 | 可变 | **5-20** |
| **采样方式** | 随机 SDE | 确定 ODE ($\eta=0$) | SDE 或 ODE | **纯 ODE** |
| **理论复杂度** | 高 | 中 | 高 | **低** |

### 7.2 损失函数演变

```
DDPM:       L = E[ ‖ε - εθ(√ᾱ·x₀ + √(1-ᾱ)·ε, t)‖² ]
               ↓ 需要 ᾱ_t = ∏(1-β_s)，需要设计 β₁,...,β_T
               ↓ 预测噪声

DDIM:       L = E[ ‖ε - εθ(...)‖² ]        ← 同 DDPM
               ↓ 但采样时去掉随机项

Score SDE:   L = E[ λ(t)·‖sθ(x,t) - ∇log p₀ₜ(x|x₀)‖² ]
               ↓ 需要设计 g(t), λ(t)
               ↓ 预测 score

Flow Match:  L = E[ ‖uθ(tz + (1-t)ε, t) - (z - ε)‖² ]
               ↓ 无需任何超参数！
               ↓ 直接回归向量场
```

### 7.3 变量数量对比

| 方法 | 必须设计的超参数 | 辅助变量数 | 网络输出含义 |
|------|------------------|------------|-------------|
| DDPM | $\beta_1, \ldots, \beta_T$ | ~5 个 ($\alpha, \bar{\alpha}, \tilde{\beta}...$) | 预测噪声 |
| DDIM | 复用 DDPM 的 | ~6 个 (+$\eta$, $\hat{x}_0$) | 预测噪声 |
| Score SDE | $g(t), \lambda(t)$ | ~4 个 ($f, g, s$...) | 预测 score |
| **Flow Matching** | **无** | **0** | **回归向量场** |

---

## 八、为什么 Flow Matching 更好？

### 8.1 简洁性优势

**DDPM 的完整公式链：**
$$\begin{cases}
\alpha_t = 1 - \beta_t \\
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s \\
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \\
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta) \\
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \\
x_{t-1} = \mu_\theta + \sqrt{\tilde{\beta}_t}\,z
\end{cases}
$$

**Flow Matching 的完整公式链：**
$$\begin{cases}
x = tz + (1-t)\epsilon \\
\text{target} = z - \epsilon \\
\text{loss} = \|u_t^\theta(x, t) - \text{target}\|^2 \\
x_{t+h} = x_t + h \cdot u_t^\theta(x_t, t)
\end{cases}
$$

### 8.2 无超参数的优势

| 问题 | DDPM/Score SDE | Flow Matching |
|------|---------------|---------------|
| "如何选择 $\beta_t$？" | 需要实验调参 | 不存在此问题 |
| "线性还是余弦调度？" | 需要尝试 | 不需要 |
| "调度对结果影响大吗？" | 影响很大 | 无影响 |
| "不同任务需要重新设计吗？" | 通常需要 | **不需要** |

### 8.3 灵活性优势

**Flow Matching 可以轻松处理任意源分布和目标分布之间的转换：**

```python
path = OTConditionalProbabilityPath(
    p_simple = GaussianSampleable(sigma=1.0),
    p_data = ImageDatasetSampleable("path/to/data")
)

trainer.train(model, path)
```

这在 DDPM 中很难做到（需要为每对分布重新设计噪声调度）。

---

## 九、Flow Matching 的完整变量表

| 变量 | 类型 | 定义域 | 说明 |
|------|------|--------|------|
| $z$ | 数据向量 | $\mathbb{R}^d$ | 目标数据（条件变量），来自 $p_{\text{data}}$ |
| $x$ | 向量 | $\mathbb{R}^d$ | 当前位置（插值后的混合样本） |
| $t$ | 时间 | $[0, 1]$ | 插值系数（0=纯噪声, 1=纯数据） |
| $\epsilon$ | 噪声向量 | $\mathbb{R}^d$ | 标准高斯噪声 $\mathcal{N}(0, I)$ |
| $p_{\text{simple}}$ | 分布 | — | 简单分布（通常是标准高斯） |
| $p_{\text{data}}$ | 分布 | — | 数据分布 |
| $p_t(x \mid z)$ | 条件密度 | — | 给定 $z$ 时 $x$ 在时刻 $t$ 的分布 |
| $u_t^{\text{target}}(x \mid z)$ | 向量场 | $\mathbb{R}^d \times [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ | 真实的条件向量场（解析可得） |
| $u_t^\theta(x, t)$ | 神经网络 | $\mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ | 学习到的向量场 |
| $\theta$ | 参数集 | — | 神经网络的全部可学习参数 |

---

## 十、Flow Matching 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **极简公式** | 无需 $\alpha$, $\beta$, $\bar{\alpha}$ 等辅助变量 |
| **无超参数** | 不需要设计噪声调度 |
| **通用性强** | 支持任意源/目标分布对 |
| **速度快** | 5-20 步即可高质量生成 |
| **确定性** | 相同输入 → 相同输出 |
| **易于实现** | 几十行代码即可完成训练 |
| **理论优美** | 条件损失 = 边际损失的隐式优化 |
| **统一框架** | DDPM/Score SDE 是 FM 的特例（VP-CFM） |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **相对较新** | 生态系统不如 DDPM 成熟 |
| **多样性有限** | 确定性输出可能缺乏多样性（可通过加噪声缓解） |
| **直线假设** | 默认线性插值可能不是最优路径（Rectified Flow 改进此问题） |
| **t→1 数值问题** | 向量场在 $t \to 1$ 时趋向无穷，需要特殊处理 |

---

## 十一、实践代码示例

### 11.1 Flow Matching 训练（OT-CFM）

```python
import torch
import torch.nn as nn

class VelocityNet(nn.Module):
    def __init__(self, dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t):
        t_emb = t.view(-1, 1)
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)


class OTCFMTrainer:
    def __init__(self, velocity_net, lr=1e-4):
        self.velocity_net = velocity_net
        self.optimizer = torch.optim.Adam(velocity_net.parameters(), lr=lr)

    def train_step(self, z):
        """
        z: 真实数据样本, shape (B, D)
        """
        B, D = z.shape
        device = z.device

        t = torch.rand(B, device=device)

        eps = torch.randn_like(z)

        x = t.view(-1, 1) * z + (1 - t.view(-1, 1)) * eps

        target = z - eps

        pred = self.velocity_net(x, t)

        loss = torch.mean((pred - target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### 11.2 Euler 采样

```python
@torch.no_grad()
def sample_euler(velocity_net, shape, N=20, device='cuda'):
    """
    Euler 方法求解 ODE 进行采样
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / N

    for i in range(N):
        t = torch.full((shape[0],), i * dt, device=device)
        v = velocity_net(x, t)
        x = x + v * dt

    return x
```

### 11.3 高阶 ODE 求解器（Runge-Kutta 4）

```python
@torch.no_grad()
def sample_rk4(velocity_net, shape, N=10, device='cuda'):
    """
    RK4 方法求解 ODE，更高精度，更少步数
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / N

    for i in range(N):
        t = torch.full((shape[0],), i * dt, device=device)
        t_half = torch.full((shape[0],), (i + 0.5) * dt, device=device)
        t_next = torch.full((shape[0],), (i + 1) * dt, device=device)

        k1 = velocity_net(x, t)
        k2 = velocity_net(x + 0.5 * dt * k1, t_half)
        k3 = velocity_net(x + 0.5 * dt * k2, t_half)
        k4 = velocity_net(x + dt * k3, t_next)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x
```

---

## 十二、四代演进全景总结

### 12.1 发展脉络图

```
2020                          2021                           2021                         2022+
  DDPM                          DDIM                        Score SDE                     Flow Matching
  ┌─────────┐                 ┌─────────┐                  ┌──────────┐                ┌──────────────┐
  │离散马尔可夫│                │非马尔可夫│                   │连续SDE/ODE│                │连续ODE       │
  │         │                 │         │                  │          │                │              │
  │L=‖ε-εθ‖²│                 │L=‖ε-εθ‖²│                  │L=λ‖s-∇logp‖²│               │L=‖u-u*‖²    │
  │         │                 │         │                  │          │                │              │
  │学:噪声   │                 │学:噪声   │                  │学:score   │                │学:向量场     │
  │         │                 │         │                  │          │                │              │
  │1000步   │  去掉随机       │50步     │  统一+连续化       │可变步数   │  彻底简化        │5-20步        │
  │         │  +加速          │         │                  │          │  +去超参数        │              │
  │设计β_t  │                 │复用β_t  │                  │设计g(t)   │                │无！          │
  └─────────┘                 └─────────┘                  └──────────┘                └──────────────┘


  开创扩散模型时代               加速采样                    统一理论框架                   主流范式
```

### 12.2 核心改进总结

| 迭代 | 核心改进 | 本质变化 | 公式行数 |
|------|----------|----------|----------|
| **DDPM → DDIM** | 去掉随机项 $\sigma_t z$ | SDE → ODE | 减少 ~30% |
| **DDPM → Score** | 统一到 SDE 框架 | 离散 → 连续 | 增加（更抽象） |
| **Score → Flow** | 直接回归向量场 | 复杂 → **简洁** | **减少 ~70%** |
| **DDPM → Flow** | 全部改进 | 最简最优 | **减少 ~80%** |

### 12.3 一句话总结四代演进

> **DDPM 证明了扩散模型可行 → DDIM 发现可以去掉随机性加速 → Score SDE 统一了所有方法的理论基础 → Flow Matching 将一切简化为最纯粹的回归问题。**
>
> **每一次迭代的本质都是：让公式更短、超参数更少、速度更快。**
