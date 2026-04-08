# 第三代：Score SDE (Score-based Generative Models via SDE)

> **论文：** Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., ICLR 2021)
>
> **地位：** 将 DDPM 和 Score-based 模型统一到 SDE 框架下，揭示了扩散模型与 ODE/SDE 的深层联系

---

## 目录

1. [核心思想](#一核心思想)
2. [从 DDPM 到 SDE 的推导](#二从-ddpm-到-sde-的推导)
3. [三种标准 SDE 形式](#三三种标准-sde-形式)
4. [反向 SDE 与 Anderson 定理](#四反向-sde-与-anderson-定理)
5. [Fokker-Planck 方程与概率流 ODE](#五fokker-planck-方程与概率流-ode)
6. [Score Matching 训练方法](#六score-matching-训练方法)
7. [采样方法](#七采样方法)
8. [与前两代的详细对比](#八与前两代的详细对比)
9. [Score SDE 的完整变量表](#九score-sde-的完整变量表)
10. [Score SDE 的优缺点总结](#十score-sde-的优缺点总结)
11. [实践代码示例](#十一实践代码示例)
12. [三代演进脉络](#十二三代演进脉络)

---

## 一、核心思想

### 1.1 从离散到连续

```
DDPM / DDIM:                    Score SDE:
┌──────────────────────┐        ┌──────────────────────────┐
│ 离散时间步 t = 0,1,...,T │        │ 连续时间 t ∈ [0, T]       │
│                      │        │                          │
│ q(x_t|x_{t-1})       │   统一  │ dx = f(x,t)dt + g(t)dW    │
│ = N(√(1-β)x, βI)     │  ──→   │                          │
│                      │        │ f(x,t): 漂移项            │
│ 需要设计 β₁,...,β_T   │        │ g(t): 扩散系数            │
└──────────────────────┘        └──────────────────────────┘
 离散的噪声调度                    连续的微分方程
```

### 1.2 什么是 Score 函数？

$$\boxed{s(x, t) := \nabla_x \log p_t(x)}$$

| 符号 | 含义 |
|------|------|
| $\nabla_x$ | 对 $x$ 的梯度（向量） |
| $p_t(x)$ | 时间 $t$ 处的概率密度 |
| $s(x, t)$ | **Score 函数**——指向概率密度增长最快的方向 |

**直觉：**
- 如果 $p_t(x)$ 是一座山的高度图
- $s(x, t) = \nabla_x \log p_t(x)$ 就是每个位置的"上山方向"
- **沿着 score 方向走 → 走向高概率区域（真实数据）**

### 1.3 Score 函数与噪声预测的关系

在 VP-SDE（DDPM 的连续版本）中，Score 函数和噪声预测有简单的关系：

$$s_\theta(x, t) = -\frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar{\alpha}(t)}}$$

| 关系 | 说明 |
|------|------|
| **Score = -Noise / 缩放因子** | 两者只差一个常数缩放 |
| **Score 更通用** | 可以处理任意 SDE，不限于 VP 类型 |
| **Noise 更直观** | "预测加了多少噪声"更容易理解 |

---

## 二、从 DDPM 到 SDE 的推导

### 2.1 回顾 DDPM 的前向过程

DDPM 前向过程（离散）：

$$x_{t} = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

重写为增量形式：

$$x_{t} - x_{t-1} = (\sqrt{1-\beta_t} - 1) x_{t-1} + \sqrt{\beta_t} \epsilon$$

### 2.2 连续化极限

令 $\Delta t = 1/T$（每步的时间间隔），则：

$$\frac{x_{t} - x_{t-1}}{\Delta t} = T(\sqrt{1-\beta_t} - 1) x_{t-1} + T\sqrt{\beta_t} \epsilon$$

当 $T \to \infty$（步数趋近无穷），$\beta_t \to 0$（每步噪声趋近于 0）。利用泰勒展开（$\beta_t$ 很小时）：

$$\sqrt{1-\beta_t} \approx 1 - \frac{\beta_t}{2}$$

因此 $\sqrt{1-\beta_t} - 1 \approx -\frac{\beta_t}{2}$。

定义连续时间的噪声调度 $\beta(t) = \lim_{T\to\infty} T \cdot \beta_{\lfloor tT \rfloor}$（注意这里乘以 $T$ 是因为时间间隔 $\Delta t = 1/T$，需要补偿）。

将增量方程取极限：

$$\frac{\Delta x}{\Delta t} \approx -\frac{1}{2}\beta(t) x + \sqrt{\beta(t)} \cdot \frac{\epsilon}{\sqrt{\Delta t}}$$

当 $\Delta t \to 0$ 时，$\frac{\epsilon}{\sqrt{\Delta t}}$ 收敛到布朗运动的增量 $\mathrm{d}W_t$（因为 $\mathrm{Var}(\epsilon/\sqrt{\Delta t}) = I/\Delta t \cdot \Delta t = I$）。

得到：

$$\mathrm{d}x = \underbrace{-\frac{1}{2}\beta(t)x}_{f(x,t)}\,\mathrm{d}t + \underbrace{\sqrt{\beta(t)}}_{g(t)}\,\mathrm{d}W_t$$

这就是 **VP-SDE (Variance Preserving SDE)**！

---

## 三、三种标准 SDE 形式

### 3.1 VP-SDE (Variance Preserving)

$$\mathrm{d}x = -\frac{1}{2}\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t$$

| 性质 | 说明 |
|------|------|
| 来源 | DDPM 的连续极限 |
| 特点 | 漂移项向原点收缩，保持方差结构 |
| 边缘分布 | $p_{0t}(x\|x_0) = \mathcal{N}(x; e^{-\frac{1}{2}\int_0^t \beta(s)ds}x_0, (1-e^{-\int_0^t \beta(s)ds})I)$ |
| 方差 | $\text{Var}(x_t\|x_0) = (1-e^{-\int_0^t \beta(s)ds})I \leq I$，方差有上界 |

**为什么叫"方差保持"？** 因为如果 $x_0$ 的方差是 $I$，那么 $x_t$ 的方差也是 $I$（无条件方差不变）。漂移项 $-\frac{1}{2}\beta(t)x$ 的"收缩"恰好抵消了扩散项的"膨胀"。

### 3.2 VE-SDE (Variance Exploding)

$$\mathrm{d}x = \underbrace{0}_{f(x,t)} \cdot \mathrm{d}t + g(t)\,\mathrm{d}W_t$$

即：

$$\mathrm{d}x = g(t)\,\mathrm{d}W_t$$

| 性质 | 说明 |
|------|------|
| 来源 | Score-based 模型（NCSN）的连续极限 |
| 特点 | **没有漂移项**，只有扩散项，方差随时间爆炸式增长 |
| 边缘分布 | $p_{0t}(x\|x_0) = \mathcal{N}(x; x_0, (\int_0^t g^2(s)ds)I)$ |
| 方差 | $\text{Var}(x_t\|x_0) = (\int_0^t g^2(s)ds)I$，随 $t$ 单调递增，无上界 |

**为什么叫"方差爆炸"？** 因为没有向心力（漂移项），噪声不断累积，方差趋向无穷。常用 $g(t)$ 选择：$g(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$（指数增长）。

**VP-SDE vs VE-SDE 的直觉对比：**

```
VP-SDE:                          VE-SDE:
x₀ ──收缩+扩散──→ x_T            x₀ ──纯扩散──→ x_T
     ↑ 漂移项拉向原点                  ↑ 没有收缩力
     ↑ 方差保持有界                     ↑ 方差无限增长
     ↑ 像弹簧+布朗运动                 ↑ 像纯布朗运动
```

### 3.3 sub-VP-SDE

$$\mathrm{d}x = -\frac{1}{2}\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}\,\mathrm{d}W_t$$

| 性质 | 说明 |
|------|------|
| 来源 | VP-SDE 的变体，Song et al. 2021 提出 |
| 特点 | 扩散系数比 VP-SDE 更大，方差始终小于等于 VP-SDE |
| 方差 | $\text{Var}(x_t\|x_0) = (1-e^{-2\int_0^t \beta(s)ds})^2 I$ |
| 实践 | 通常效果不如 VP-SDE 和 VE-SDE |

---

## 四、反向 SDE 与 Anderson 定理

### 4.1 Anderson (1982) 定理

**定理（Anderson, 1982）：** 给定前向 SDE：

$$\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}W_t$$

其反向 SDE 为：

$$\boxed{\mathrm{d}x = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]\,\mathrm{d}t + g(t)\,\overline{\mathrm{d}W}_t}$$

其中 $\overline{\mathrm{d}W}_t$ 是**反向布朗运动**（时间倒流的维纳过程）。

### 4.2 推导思路（非严格）

**核心思想：** 利用贝叶斯定理将前向转移概率"反转"。

**Step 1：** 前向转移概率（短时间 $\Delta t$）：

$$p(x_{t+\Delta t}|x_t) \approx \mathcal{N}(x_{t+\Delta t}; x_t + f(x_t,t)\Delta t, g(t)^2 \Delta t \cdot I)$$

**Step 2：** 反向转移概率（贝叶斯定理）：

$$p(x_t|x_{t+\Delta t}) = \frac{p(x_{t+\Delta t}|x_t) p_t(x_t)}{p_{t+\Delta t}(x_{t+\Delta t})}$$

**Step 3：** 取对数并对 $x_t$ 展开（类似 Laplace 近似）：

$$\log p(x_t|x_{t+\Delta t}) \approx \log p(x_{t+\Delta t}|x_t) + \log p_t(x_t) - \log p_{t+\Delta t}(x_{t+\Delta t})$$

**Step 4：** 对 $\log p_t(x_t)$ 在 $x_{t+\Delta t}$ 处泰勒展开：

$$\log p_t(x_t) \approx \log p_t(x_{t+\Delta t}) + \nabla_x \log p_t(x_{t+\Delta t}) \cdot (x_t - x_{t+\Delta t})$$

**Step 5：** 将所有项合并，配方后得到反向转移分布是高斯的，其均值为：

$$x_{t+\Delta t} + \left[f(x_{t+\Delta t}, t) - g(t)^2 \nabla_x \log p_t(x_{t+\Delta t})\right] \Delta t$$

方差为 $g(t)^2 \Delta t \cdot I$。

取 $\Delta t \to 0$ 的极限，就得到反向 SDE。

### 4.3 关键洞察

> 🎯 **反向 SDE 中需要知道 $\nabla_x \log p_t(x)$ —— 这就是神经网络要学习的东西！**

一旦我们学到了 score 函数 $s_\theta(x,t) \approx \nabla_x \log p_t(x)$，就可以用数值方法求解反向 SDE 进行采样。

---

## 五、Fokker-Planck 方程与概率流 ODE

### 5.1 Fokker-Planck 方程

Fokker-Planck 方程描述了 SDE 驱动下概率密度随时间的演化：

$$\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot \left[f(x,t) p_t(x)\right] + \frac{1}{2}\nabla_x^2 \cdot \left[g(t)^2 p_t(x)\right]$$

| 项 | 含义 |
|------|------|
| $-\nabla_x \cdot [f \cdot p_t]$ | 漂移项引起的概率流 |
| $\frac{1}{2}\nabla_x^2 \cdot [g^2 \cdot p_t]$ | 扩散项引起的概率扩散 |

**直觉：** Fokker-Planck 方程是概率密度版的"连续性方程"——它描述了概率质量如何随时间重新分配。

### 5.2 概率流 ODE 的推导

**核心发现：** 每个 SDE 都对应一个**确定性 ODE**，其边缘分布与 SDE 完全相同！

**推导：** 将 Fokker-Planck 方程改写：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot \left[\left(f - \frac{1}{2}g^2 \nabla_x \log p_t\right) p_t\right]$$

这个形式可以看作是某个 ODE 的连续性方程：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot \left[\tilde{f}(x,t) \cdot p_t\right]$$

其中 $\tilde{f}(x,t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$。

因此，对应的 ODE 为：

$$\boxed{\mathrm{d}x = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]\mathrm{d}t}$$

**这就是概率流 ODE（Probability Flow ODE）！**

### 5.3 SDE vs 概率流 ODE 对比

| 性质 | 反向 SDE | 概率流 ODE |
|------|----------|-----------|
| 公式 | $\mathrm{d}x = [f - g^2\nabla\log p_t]\mathrm{d}t + g\,\overline{\mathrm{d}W}_t$ | $\mathrm{d}x = [f - \frac{1}{2}g^2\nabla\log p_t]\mathrm{d}t$ |
| 随机性 | 有（布朗运动项） | **无**（纯确定性） |
| 边缘分布 | $p_t(x)$ | **相同的** $p_t(x)$ |
| 采样多样性 | 高 | 低（确定性映射） |
| 对应关系 | DDPM（η=1） | DDIM（η=0） |

**注意系数差异：** SDE 中 score 的系数是 $-g^2$，ODE 中是 $-\frac{1}{2}g^2$。这是因为 ODE 没有"扩散"来补充随机性，所以需要减半 score 的贡献来保持边缘分布不变。

---

## 六、Score Matching 训练方法

### 6.1 为什么学 Score？

**核心观察：** 给定 SDE 的漂移项和扩散项，如果我们知道 score 函数 $\nabla_x \log p_t(x)$，就可以构造反向 SDE 或概率流 ODE 进行采样。

**问题：** 我们不知道真实的 $p_t(x)$，所以无法直接计算 $\nabla_x \log p_t(x)$。

**解决方案：** 训练一个神经网络 $s_\theta(x,t)$ 来近似 score 函数。

### 6.2 显式 Score Matching（不可行）

最直接的目标是：

$$\mathcal{L}_{\text{ESM}} = \mathbb{E}_{t}\left[\lambda(t) \mathbb{E}_{p_t(x)}\left[\|s_\theta(x,t) - \nabla_x \log p_t(x)\|^2\right]\right]$$

**问题：** 需要知道 $\nabla_x \log p_t(x)$，而这正是我们想学的！

### 6.3 Denoising Score Matching（可行！）

**定理（Vincent, 2011）：** 以下目标与显式 Score Matching 的梯度相同：

$$\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{t}\left[\lambda(t) \mathbb{E}_{x_0 \sim p_{\text{data}},\, x_t \sim p_{0t}(x_t|x_0)}\left[\left\|s_\theta(x_t, t) - \nabla_{x_t} \log p_{0t}(x_t|x_0)\right\|^2\right]\right]}$$

**关键区别：**
- 显式 SM 需要计算 $\nabla_x \log p_t(x)$（边际分布的 score，不可计算）
- 去噪 SM 只需要计算 $\nabla_{x_t} \log p_{0t}(x_t|x_0)$（条件分布的 score，**可以解析计算**！）

**为什么等价？** 直觉上，学习"从带噪数据中恢复干净数据的梯度方向"等价于学习"概率密度增长最快的方向"，因为条件分布 $p_{0t}(x_t|x_0)$ 的 score 指向数据 $x_0$ 的方向。

### 6.4 条件 Score 的解析计算

对于 VP-SDE，$p_{0t}(x_t|x_0) = \mathcal{N}(x_t; \mu_t(x_0), \sigma_t^2 I)$，其中 $\mu_t = e^{-\frac{1}{2}\int_0^t \beta(s)ds}x_0$，$\sigma_t^2 = 1 - e^{-\int_0^t \beta(s)ds}$。

高斯分布的 score：

$$\nabla_{x_t} \log p_{0t}(x_t|x_0) = -\frac{x_t - \mu_t(x_0)}{\sigma_t^2} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}(t)}}$$

其中 $\epsilon$ 是添加的噪声（因为 $x_t = \mu_t x_0 + \sigma_t \epsilon$）。

**代入损失函数：**

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\lambda(t)\left\|s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}(t)}}\right\|^2\right]$$

如果令 $s_\theta(x,t) = -\frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar{\alpha}(t)}}$，则：

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\frac{\lambda(t)}{1-\bar{\alpha}(t)}\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|^2\right]$$

**当 $\lambda(t) = 1-\bar{\alpha}(t)$ 时，这恰好就是 DDPM 的简化损失！**

### 6.5 权重函数 λ(t) 的选择

| 选择 | 公式 | 效果 |
|------|------|------|
| $\lambda(t) = 1-\bar{\alpha}(t)$ | VP-SDE 默认 | 等价于 DDPM 的 L_simple |
| $\lambda(t) = \sigma_t^2$ | 与噪声方差成比例 | 更关注低噪声区域 |
| $\lambda(t) = 1$ | 均匀权重 | 所有时间步同等重要 |
| $\lambda(t) = g(t)^2$ | Song et al. 推荐 | 理论最优权重 |

---

## 七、采样方法

### 7.1 随机采样器（SDE Solver）

**算法：Euler-Maruyama 方法求解反向 SDE**

| 步骤 | 操作 |
|------|------|
| **输入** | Score 网络 $s_\theta$，SDE 参数 $f, g$，步数 $N$ |
| **初始化** | $x_N \sim p_{\text{prior}}$（通常是 $\mathcal{N}(0, I)$），$h = T/N$ |
| **循环** | 对 $i = N-1, N-2, \ldots, 0$： |
| ① | 计算 drift: $\hat{f} = f(x_i, t_i) - g(t_i)^2 s_\theta(x_i, t_i)$ |
| ② | 采样 $\epsilon \sim \mathcal{N}(0, I)$ |
| ③ | $x_{i-1} = x_i + h \cdot \hat{f} + g(t_i)\sqrt{h} \cdot \epsilon$ |
| **输出** | $x_0$ |

### 7.2 确定性采样器（ODE Solver / Probability Flow ODE）

**算法：Euler 方法求解概率流 ODE**

| 步骤 | 操作 |
|------|------|
| **输入** | Score 网络 $s_\theta$，SDE 参数 $f, g$，步数 $N$ |
| **初始化** | $x_N \sim p_{\text{prior}}$，$h = T/N$ |
| **循环** | 对 $i = N-1, N-2, \ldots, 0$： |
| ① | 计算 drift: $\hat{f} = f(x_i, t_i) - \frac{1}{2}g(t_i)^2 s_\theta(x_i, t_i)$ |
| ② | $x_{i-1} = x_i + h \cdot \hat{f}$ |
| **输出** | $x_0$ |

> ⚠️ 注意与随机采样的区别：**没有 $g(t)\sqrt{h}\cdot\epsilon$ 项！** 而且 score 的系数是 $\frac{1}{2}g^2$ 而非 $g^2$。

### 7.3 预测校正采样器（Predictor-Corrector）

结合两者优势的高级采样器：

```
for each time step:
    Predictor: 用 ODE/SDE 大步前进（快速移动到下一时刻）
    Corrector: 用 Langevin MCMC 小步修正（提高当前时刻的采样精度）
```

**具体算法：**

| 步骤 | 操作 |
|------|------|
| **输入** | Score 网络 $s_\theta$，SDE 参数 $f, g$，校正步数 $M$，步长 $r$ |
| **初始化** | $x \sim p_{\text{prior}}$ |
| **循环** | 对每个时间步 $t_i$： |
| **Predictor** | $x \leftarrow x + h[f(x,t_i) - g(t_i)^2 s_\theta(x,t_i)] + g(t_i)\sqrt{h}\epsilon$ |
| **Corrector** | 重复 $M$ 次：$x \leftarrow x + r \cdot s_\theta(x, t_i) + \sqrt{2r}\epsilon'$ |
| **输出** | $x_0$ |

**Corrector 的原理：** Langevin MCMC 采样。给定 score 函数，Langevin 动力学可以从 $p_t(x)$ 中精确采样：

$$x \leftarrow x + r \cdot \nabla_x \log p_t(x) + \sqrt{2r} \cdot \epsilon$$

多步迭代后，$x$ 的分布收敛到 $p_t(x)$。

---

## 八、与前两代的详细对比

### 8.1 公式对比总表

| 项目 | DDPM (2020) | DDIM (2021) | Score SDE (2021) |
|------|-------------|-------------|------------------|
| **数学框架** | 离散马尔可夫链 | 离散非马尔可夫 | **连续 SDE/ODE** |
| **前向过程** | $q(x_t\|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | 同左 | $\mathrm{d}x = f(x,t)\mathrm{d}t + g(t)\mathrm{d}W_t$ |
| **学习目标** | 噪声预测 $\epsilon_\theta$ | 噪声预测 $\epsilon_\theta$ | **Score 预测 $s_\theta = \nabla_x \log p_t(x)$** |
| **损失函数** | $\|\epsilon - \epsilon_\theta\|^2$ | 同左 | $\lambda(t)\|s_\theta - \nabla \log p_{0t}\|^2$ |
| **噪声调度** | $\beta_t$（超参数） | 复用 DDPM 的 | $g(t)$ 或 $\beta(t)$（仍需设计） |
| **采样方式** | 随机（1000步） | 确定/可选（50步） | 随机或确定（可变步数） |
| **理论统一性** | 单一模型 | DDPM 变体 | **统一所有扩散模型** |

### 8.2 关键参数对比

| 参数 | DDPM | DDIM | Score SDE |
|------|------|------|-----------|
| 时间表示 | 离散 $t \in \{1,\ldots,T\}$ | 同左 | **连续 $t \in [0,T]$** |
| 噪声调度 | $\beta_t \in (0,1)$ | 同左 | $g(t): [0,T] \to \mathbb{R}^+$ |
| 网络输出 | $\hat{\epsilon} \in \mathbb{R}^d$ | 同左 | **$\hat{s} \in \mathbb{R}^d$（score 向量）** |
| 控制参数 | 无 | $\eta \in [0,1]$ | $\lambda(t)$（加权函数） |
| 扩散系数 | $\sigma_t$（固定或 $\tilde{\beta}_t$） | 被 $\eta$ 取代 | **$g(t)$（连续函数）** |

---

## 九、Score SDE 的完整变量表

| 变量 | 类型 | 定义域 | 说明 |
|------|------|--------|------|
| $x(t)$ | 数据向量 | $\mathbb{R}^d$ | 连续时间 $t$ 时的数据状态 |
| $t$ | 连续时间 | $[0, T]$ | 当前时刻（通常 $T=1$） |
| $f(x,t)$ | 漂移场 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | SDE 的确定性部分 |
| $g(t)$ | 扩散系数 | $[0,T] \to \mathbb{R}^+$ | 控制 SDE 随机性强度 |
| $W_t$ | 布朗运动 | — | 标准维纳过程 |
| $s(x,t)$ | Score 函数 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | $\nabla_x \log p_t(x)$ |
| $s_\theta(x,t)$ | 神经网络 | $\mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ | Score 的近似 |
| $\lambda(t)$ | 权重函数 | $[0,T] \to \mathbb{R}^+$ | 不同时间步的损失权重 |
| $p_t(x)$ | 概率密度 | $\mathbb{R}^d \to \mathbb{R}^+$ | 时间 $t$ 时的数据分布 |
| $p_{0t}(x\|x_0)$ | 条件密度 | — | 给定 $x_0$ 时 $x(t)$ 的分布 |
| $\beta(t)$ | 噪声调度 | $[0,T] \to \mathbb{R}^+$ | VP-SDE 的噪声方差率 |
| $\bar{\alpha}(t)$ | 累积保留 | $[0,T] \to (0,1]$ | $e^{-\int_0^t \beta(s)ds}$（VP-SDE） |

---

## 十、Score SDE 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **理论统一** | 将 DDPM、Score-based、Langevin MCMC 统一到一个框架 |
| **灵活性强** | 支持多种 SDE 类型（VP、VE、sub-VP 等） |
| **采样多样** | 同时支持随机采样和确定性采样 |
| **可控精度** | 可通过 predictor-corrector 平衡速度和质量 |
| **精确对数似然** | 概率流 ODE 允许精确计算对数似然 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **仍需设计 $g(t)$** | 扩散系数仍是预设的超参数 |
| **公式更抽象** | SDE 比离散链更难理解 |
| **训练未简化** | 本质上还是 Score Matching / Noise Prediction |
| **实现复杂度高** | 需要数值积分 SDE/ODE |
| **SDE 求解器选择** | 不同的数值方法（Euler、RK 等）影响采样质量 |

---

## 十一、实践代码示例

### 11.1 Score SDE 训练（VP-SDE）

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, channels=3, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_input = torch.cat([x, t_emb], dim=1)
        return self.net(x_input)


class VPSDETrainer:
    def __init__(self, score_net, beta_min=0.1, beta_max=20.0, T=1.0):
        self.score_net = score_net
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t):
        return torch.exp(-0.5 * t * (2 * self.beta_min + t * (self.beta_max - self.beta_min)))

    def train_step(self, x_0):
        t = torch.rand(x_0.shape[0], device=x_0.device) * self.T
        eps = torch.randn_like(x_0)

        ab = self.alpha_bar(t).view(-1, 1, 1, 1)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * eps

        score_pred = self.score_net(x_t, t)

        target_score = -eps / torch.sqrt(1 - ab)

        loss = torch.mean((score_pred - target_score) ** 2)
        return loss
```

### 11.2 概率流 ODE 采样

```python
@torch.no_grad()
def sample_pf_ode(score_net, shape, beta_min=0.1, beta_max=20.0, T=1.0, N=100, device='cuda'):
    dt = T / N
    x = torch.randn(shape, device=device)

    for i in range(N, 0, -1):
        t = torch.full((shape[0],), i * dt, device=device)
        beta_t = beta_min + t * (beta_max - beta_min)
        ab = torch.exp(-0.5 * t * (2 * beta_min + t * (beta_max - beta_min)))

        score = score_net(x, t)

        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x - 0.5 * beta_t.view(-1, 1, 1, 1) * (1 - ab.view(-1, 1, 1, 1)) * score

        x = x - drift * dt

    return x
```

### 11.3 反向 SDE 采样

```python
@torch.no_grad()
def sample_reverse_sde(score_net, shape, beta_min=0.1, beta_max=20.0, T=1.0, N=100, device='cuda'):
    dt = T / N
    x = torch.randn(shape, device=device)

    for i in range(N, 0, -1):
        t = torch.full((shape[0],), i * dt, device=device)
        beta_t = beta_min + t * (beta_max - beta_min)
        ab = torch.exp(-0.5 * t * (2 * beta_min + t * (beta_max - beta_min)))

        score = score_net(x, t)

        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x - beta_t.view(-1, 1, 1, 1) * (1 - ab.view(-1, 1, 1, 1)) * score
        diffusion = torch.sqrt(beta_t).view(-1, 1, 1, 1)

        eps = torch.randn_like(x)
        x = x - drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * eps

    return x
```

---

## 十二、三代演进脉络

```
DDPM (2020):                        DDIM (2021):                     Score SDE (2021):
┌────────────────┐                  ┌────────────────┐              ┌──────────────────────┐
│ 离散马尔可夫链  │                  │ 离散非马尔可夫  │              │ 连续 SDE/ODE         │
│                │                  │                │              │                      │
│ L = ‖ε - εθ‖²  │                  │ L = ‖ε - εθ‖²  │              │ L = λ·‖sθ - ∇log p‖²│
│                │                  │                │              │                      │
│ 学: εθ (噪声)   │                  │ 学: εθ (噪声)   │              │ 学: sθ (score)       │
│                │                  │                │              │                      │
│ 1000 步采样    │  去掉随机+加速  │ 50 步采样      │  统一+连续化  │ 可变步数采样          │
│                │                  │                │              │                      │
│ 设计 β_t       │                  │ 复用 β_t       │              │ 设计 g(t)             │
└────────────────┘                  └────────────────┘              └──────────────────────┘
  开创者                              加速者                            统一者
```

### 核心贡献总结

1. **DDPM** 证明了扩散模型可行，但采样慢、公式复杂
2. **DDIM** 发现随机性是可选的，去掉后变成 ODE，可跳步加速
3. **Score SDE** 将一切统一到连续 SDE 框架，揭示 DDPM 和 DDIM 是 SDE/ODE 的离散化特例

> **下一章：Flow Matching —— 如何彻底摆脱噪声调度的束缚，实现最简洁的训练范式？**
