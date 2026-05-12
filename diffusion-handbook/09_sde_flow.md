# 第9章 SDE 统一框架与 Flow Matching

> 从离散到连续，从特殊到一般——SDE 框架将 DDPM 和 Score-Based 模型统一到同一个数学语言下。而 Flow Matching 则代表了最新的范式转换：不再需要设计噪声调度，只需要回归一个向量场。

---

## 9.1 从离散到连续：SDE 统一框架

### 9.1.1 动机

DDPM 的前向过程是离散的：$t = 0, 1, 2, \ldots, T$。但物理中的扩散现象（如墨水在水中扩散）是连续的。如果我们让步长 $\Delta t \to 0$，步数 $T \to \infty$，离散过程会趋向什么？

答案是一个**随机微分方程（SDE）**。

### 9.1.2 从 DDPM 到 SDE 的推导

DDPM 前向过程：

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$

改写为增量形式：

$$\Delta x = x_t - x_{t-1} = (\sqrt{1-\beta_t} - 1) x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$

当 $\beta_t$ 很小时，$\sqrt{1-\beta_t} \approx 1 - \frac{\beta_t}{2}$（泰勒展开），所以：

$$\Delta x \approx -\frac{\beta_t}{2} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}$$

令 $\Delta t = 1/T$，$\beta_t = \beta(t)\Delta t$，当 $T \to \infty$：

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW$$

这就是 **VP-SDE（Variance Preserving SDE）**！

### 9.1.3 三种标准 SDE

| SDE 类型 | 前向方程 | 特点 |
|----------|---------|------|
| VP-SDE | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW$ | 方差有界，对应 DDPM |
| VE-SDE | $dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dW$ | 方差爆炸，对应 NCSN |
| Sub-VP-SDE | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})}\,dW$ | VP 的改进版 |

**VP-SDE 的直觉**：信号以速率 $\frac{1}{2}\beta(t)$ 衰减，同时以速率 $\beta(t)$ 加入噪声。"方差保持"是因为信号衰减释放的"空间"恰好被噪声填满，总方差保持有界。

**VE-SDE 的直觉**：只加噪声，不衰减信号。方差随时间增长（"爆炸"），但数据分布的支撑集也在扩大，有利于覆盖更多模式。

---

## 9.2 反向 SDE 与 Anderson 定理

### 9.2.1 Anderson 定理

给定前向 SDE $dx = f(x, t)dt + g(t)dW$，其反向 SDE 为：

$$dx = \left[f(x, t) - g^2(t) \nabla_x \log p_t(x)\right] dt + g(t) d\bar{W}$$

其中 $\bar{W}$ 是反向时间的布朗运动。

**这个定理的意义**：只要知道 Score 函数 $\nabla_x \log p_t(x)$，就能写出反向 SDE，从噪声采样出数据。

### 9.2.2 VP-SDE 的反向 SDE

代入 $f(x,t) = -\frac{1}{2}\beta(t)x$，$g(t) = \sqrt{\beta(t)}$：

$$dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)} d\bar{W}$$

**与 DDPM 的对应**：
- $-\frac{1}{2}\beta(t)x$：信号衰减项
- $-\beta(t)\nabla_x \log p_t(x)$：Score 引导项（指向数据方向）
- $\sqrt{\beta(t)} d\bar{W}$：随机噪声项

当离散化时，这就是 DDPM 的采样公式！

---

## 9.3 Fokker-Planck 方程与概率流 ODE

### 9.3.1 Fokker-Planck 方程

Fokker-Planck 方程描述概率密度 $p(x, t)$ 如何随时间演化：

$$\frac{\partial p(x, t)}{\partial t} = -\nabla_x \cdot [f(x, t) p(x, t)] + \frac{1}{2}\nabla_x^2 [g^2(t) p(x, t)]$$

**物理意义**：
- 第一项：漂移项——概率密度沿着漂移方向"流动"
- 第二项：扩散项——概率密度向四周"扩散"

### 9.3.2 概率流 ODE

**关键发现**：存在一个确定性的 ODE，它的边缘分布 $p(x, t)$ 与 SDE 完全相同：

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g^2(t) \nabla_x \log p_t(x)$$

对于 VP-SDE：

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\nabla_x \log p_t(x)$$

**概率流 ODE 的意义**：
1. **确定性**：没有随机项，相同输入总是产生相同输出
2. **可逆**：从 $x_0$ 可以精确恢复 $x_T$（Neural ODE 性质）
3. **精确对数似然**：可以通过变量替换公式计算
4. **DDIM 的连续版本**：DDIM 就是对概率流 ODE 的欧拉法离散化

---

## 9.4 Score Matching 训练方法

### 9.4.1 加权 Score Matching

在 SDE 框架下，训练目标是加权去噪 Score Matching：

$$\mathcal{L} = \mathbb{E}_t\left[\lambda(t) \mathbb{E}_{x_0}\mathbb{E}_{x_t|x_0}\left[\|s_\theta(x_t, t) - \nabla_{x_t}\log p_t(x_t|x_0)\|^2\right]\right]$$

其中 $\lambda(t)$ 是权重函数。不同的 $\lambda(t)$ 对应不同的优化目标：

| $\lambda(t)$ | 等价于 |
|-------------|--------|
| $\lambda(t) = g^2(t)$ | 等价于 DDPM 的 VLB |
| $\lambda(t) = 1$ | 等价于 DDPM 的简化损失 |

### 9.4.2 Score 的实际计算

对于 VP-SDE，给定 $x_0$ 时 $x_t$ 的条件分布为：

$$p_t(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}(t)}x_0, (1-\bar{\alpha}(t))I)$$

条件 Score：

$$\nabla_{x_t}\log p_t(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}(t)}x_0}{1-\bar{\alpha}(t)} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}(t)}}$$

训练时：
1. 采样 $x_0$，$t$，$\epsilon \sim \mathcal{N}(0, I)$
2. 计算 $x_t = \sqrt{\bar{\alpha}(t)}x_0 + \sqrt{1-\bar{\alpha}(t)}\epsilon$
3. 计算 Score 预测 $s_\theta(x_t, t)$
4. 计算损失 $\|s_\theta(x_t, t) + \epsilon/\sqrt{1-\bar{\alpha}(t)}\|^2$

**与 DDPM 的等价性**：将 $s_\theta = -\epsilon_\theta/\sqrt{1-\bar{\alpha}(t)}$ 代入，损失变为 $\|\epsilon - \epsilon_\theta\|^2 / (1-\bar{\alpha}(t))$，忽略分母就是 DDPM 的简化损失。

---

## 9.5 Flow Matching：新一代范式

### 9.5.1 动机：为什么需要 Flow Matching？

SDE 框架虽然统一，但仍有问题：
1. **需要设计噪声调度** $\beta(t)$ 或 $g(t)$——这是一个超参数
2. **训练目标间接**：通过 Score Matching 间接学习向量场
3. **ODE 轨迹可能弯曲**：导致数值求解需要很多步

Flow Matching 提出了一个更直接的方案：**直接回归向量场**。

### 9.5.2 连续正则化流（CNF）

CNF 用一个 ODE 将简单分布 $p_0$（如高斯）变换为复杂分布 $p_1$（如数据分布）：

$$\frac{dx}{dt} = v_t(x), \quad x(0) \sim p_0, \quad x(1) \sim p_1$$

其中 $v_t(x)$ 是**向量场**——告诉每个点在每个时刻应该往哪个方向走。

**之前的 CNF 训练方法**（FFJORD）需要计算雅可比行列式，训练不稳定且计算昂贵。

### 9.5.3 条件流匹配（CFM）

Flow Matching 的核心创新：**不需要计算雅可比行列式，直接回归向量场**。

**训练目标**：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[\|v_\theta(x, t) - u_t(x|x_0)\|^2\right]$$

其中 $u_t(x|x_0)$ 是**条件向量场**——从 $x_0$ 出发的最优路径上的速度。

### 9.5.4 最优传输路径

最简单的条件向量场是**最优传输（OT）路径**——从噪声到数据的最短路径（直线）：

$$x_t = (1-t) x_0 + t x_1, \quad x_0 \sim \mathcal{N}(0, I), \quad x_1 \sim p_{\text{data}}$$

条件向量场：

$$u_t(x|x_1) = x_1 - x_0$$

**直觉**：从起点 $x_0$ 沿直线走向终点 $x_1$，速度恒定为 $x_1 - x_0$。

### 9.5.5 CFM ≡ FM 的等价性

**定理**（Lipman et al., 2023）：条件流匹配的梯度等于（无条件）流匹配的梯度：

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

这意味着：**训练时只需要条件向量场（容易计算），但学到的向量场可以无条件采样（不需要知道数据分布）。**

### 9.5.6 Flow Matching 的训练算法

```
重复以下步骤：
1. 采样数据 x₁ ~ p_data
2. 采样噪声 x₀ ~ N(0, I)
3. 采样时间 t ~ Uniform(0, 1)
4. 计算中间点 x_t = (1-t)x₀ + t·x₁
5. 计算目标向量 u = x₁ - x₀
6. 计算损失 L = ‖v_θ(x_t, t) - u‖²
7. 更新参数 θ ← θ - η · ∇_θ L
```

### 9.5.7 Flow Matching 的采样

```
1. 采样 x₀ ~ N(0, I)
2. 用 ODE 求解器从 t=0 积分到 t=1：
   dx/dt = v_θ(x, t)
3. 输出 x(1)
```

### 9.5.8 Flow Matching 与扩散模型的关系

| 方面 | 扩散模型（DDPM/SDE） | Flow Matching |
|------|---------------------|---------------|
| 前向过程 | 需要设计噪声调度 | 直线插值（无需设计） |
| 训练目标 | 预测噪声或 Score | 回归向量场 |
| 反向过程 | SDE 或 ODE | ODE |
| 路径形状 | 弯曲 | 直线（OT） |
| 数值求解 | 需要很多步 | 步数更少 |

**关键区别**：Flow Matching 的 OT 路径是直线，ODE 求解器可以用更少的步数精确求解。而扩散模型的路径是弯曲的，需要更多步数。

---

## 9.6 Rectified Flow

### 9.6.1 核心思想

Liu et al. (2023) 提出了 Rectified Flow，进一步优化 Flow Matching：

1. 用直线连接噪声和数据的"配对"
2. 重新流动（reflow）：用学到的模型重新配对，让路径更直
3. 经过 1-2 次 reflow，路径几乎变成直线，1 步就能生成高质量图像

### 9.6.2 Reflow 过程

```
第 1 轮：随机配对 (x₀, x₁)，训练 v_θ
第 2 轮：用 v_θ 从 x₁ 生成对应的 x₀，重新配对，再训练
第 3 轮：重复...
```

每次 reflow 都让路径更直。理论上，经过无穷次 reflow，路径变成完美的直线。

---

## 9.7 本章小结

| 框架 | 核心方程 | 训练目标 | 采样方式 |
|------|---------|---------|---------|
| DDPM | 离散马尔可夫链 | $\Vert\epsilon - \epsilon_\theta\Vert^2$ | 随机逐步 |
| SDE | $dx = f\,dt + g\,dW$ | Score Matching | SDE/ODE |
| Flow Matching | $dx/dt = v_t(x)$ | $\Vert v - u\Vert^2$ | ODE |

**从 DDPM 到 Flow Matching 的演进**：
1. DDPM：离散 → SDE：连续 → Flow Matching：更简洁的连续
2. 预测噪声 → 预测 Score → 回归向量场
3. 弯曲路径 → 直线路径
4. 需要设计调度 → 无需设计

---

## 练习与思考

1. **SDE 推导**：验证当 $\beta_t$ 很小时，$\sqrt{1-\beta_t} \approx 1 - \beta_t/2$（泰勒展开到一阶）。

2. **概率流 ODE**：对 VP-SDE，写出概率流 ODE 的具体形式。当 $\nabla_x \log p_t(x) = -x$ 时（标准高斯分布的 Score），求解这个 ODE。

3. **Flow Matching**：设 $x_0 = (0, 0)^T$，$x_1 = (1, 2)^T$。写出 OT 路径 $x_t$ 和条件向量场 $u_t$ 的表达式。

4. **思考题**：为什么直线路径（OT）比弯曲路径需要更少的 ODE 求解步数？（提示：考虑欧拉法的误差与曲率的关系）
