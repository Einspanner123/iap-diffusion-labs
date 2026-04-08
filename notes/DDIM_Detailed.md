# 第二代：DDIM (Denoising Diffusion Implicit Models)

> **论文：** Denoising Diffusion Implicit Models (Song et al., ICLR 2021)
>
> **地位：** 发现 DDPM 可以改写为非马尔可夫过程，去掉随机项后变成 ODE，采样速度提升 20 倍

---

## 目录

1. [核心思想](#一核心思想)
2. [从 DDPM 到 DDIM 的推导](#二从-ddpm-到-ddim-的推导)
3. [DDIM 的统一采样公式](#三ddim-的统一采样公式)
4. [η 参数的核心作用](#四η-参数的核心作用)
5. [边缘分布一致性证明](#五边缘分布一致性证明)
6. [为什么可以复用 DDPM 的训练权重](#六为什么可以复用-ddpm-的训练权重)
7. [与 DDPM 的详细对比](#七与-ddpm-的详细对比)
8. [DDIM 的完整算法](#八ddim-的完整算法)
9. [DDIM 与 Neural ODE 的关系](#九ddim-与-neural-ode-的关系)
10. [DDIM 的优缺点总结](#十ddim-的优缺点总结)
11. [实践代码示例](#十一实践代码示例)
12. [DDPM → DDIM 的演进脉络](#十二ddpm--ddim-的演进脉络)

---

## 一、核心思想

### 1.1 从 DDPM 出发的问题

回顾 DDPM 的采样公式：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)}_{\text{确定性部分 } \hat{\mu}_t(x_t, t)} + \underbrace{\sigma_t z}_{\text{随机部分}}, \quad z \sim \mathcal{N}(0, I)$$

其中 $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$。

**问题：**
- 每一步都有随机项 $\sigma_t z$ → 需要走完所有 1000 步才能保证质量
- 无法跳步 → 不能加速（因为随机性是逐步累积的，跳步会丢失中间的随机性信息）
- 输出不确定 → 相同输入产生不同结果

### 1.2 DDIM 的关键洞察

> **DDPM 的 SDE 不是唯一的反向路径！存在一族非马尔可夫的反向过程，它们都共享相同的边缘分布 $q(x_t|x_0)$。**

这意味着：**我们可以选择一个"更好"的反向路径——确定性、可跳跃的 ODE。**

### 1.3 核心直觉

```
DDPM 的反向过程：每一步都依赖"前一步"（马尔可夫）
    x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0
    每步加随机噪声 z → 必须一步一步走

DDIM 的反向过程：每一步都依赖"预测的 x₀"（非马尔可夫）
    x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0
    每步从 x_t 先预测 x₀，再从 x₀ 推算 x_{t-1}
    不加随机噪声 → 可以跳步！
```

---

## 二、从 DDPM 到 DDIM 的推导

### 2.1 回顾 DDPM 的前向过程

DDPM 的前向（加噪）过程，给定 $x_0$ 时 $x_t$ 的分布：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)$$

重参数化：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**关键观察：** 这个边缘分布 $q(x_t|x_0)$ 只依赖于 $x_0$，不依赖于中间的 $x_1, \ldots, x_{t-1}$。也就是说，**无论前向过程是马尔可夫的还是非马尔可夫的，只要 $q(x_t|x_0)$ 保持不变，训练目标就不变。**

### 2.2 DDPM 假设了什么？DDIM 放宽了什么？

**DDPM 的假设（马尔可夫反向过程）：**

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

反向分布 $p_\theta(x_{t-1}|x_t)$ 只依赖 $x_t$，不依赖 $x_0$。

**DDIM 的放宽（非马尔可夫反向过程）：**

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta^{(\sigma)}(x_{t-1} | x_t, x_0)$$

反向分布 $p_\theta^{(\sigma)}(x_{t-1}|x_t, x_0)$ 可以同时依赖 $x_t$ 和 $x_0$。这里 $\sigma$ 是一个新引入的参数，控制反向过程的随机性。

**约束条件：** 新的反向过程必须保证边缘分布不变，即：

$$\int p_\theta^{(\sigma)}(x_{t-1}|x_t, x_0) \cdot q(x_t|x_0) \, \mathrm{d}x_t = q(x_{t-1}|x_0)$$

### 2.3 构造非马尔可夫前向过程

DDIM 论文的关键一步：**定义一个新的前向过程**（不是 DDPM 的马尔可夫链），使得：

1. 边缘分布 $q_\sigma(x_t|x_0)$ 与 DDPM 完全相同
2. 反向过程 $q_\sigma(x_{t-1}|x_t, x_0)$ 可以显式写出
3. 反向过程中包含一个可调参数 $\sigma$，控制随机性

**定义新的前向过程：**

给定 $x_0$ 和 $x_t$（其中 $x_t$ 服从 DDPM 的边缘分布），定义：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_\sigma(x_t, x_0, t), \sigma_t^2 I)$$

其中均值 $\tilde{\mu}_\sigma$ 和方差 $\sigma_t^2$ 需要满足边缘分布约束。

### 2.4 推导反向分布的均值

**目标：** 找到 $\tilde{\mu}_\sigma(x_t, x_0, t)$ 使得 $x_{t-1} \sim q_\sigma(x_{t-1}|x_t, x_0)$ 的边缘分布等于 $q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$。

**推导思路：** 我们知道 $x_{t-1}$ 可以写成：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_{t-1}$$

其中 $\epsilon_{t-1} \sim \mathcal{N}(0, I)$。同时，$x_t$ 可以写成：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_t$$

其中 $\epsilon_t \sim \mathcal{N}(0, I)$。

**DDIM 的构造方法：** 将 $\epsilon_{t-1}$ 分解为两部分——与 $\epsilon_t$ 相关的确定性部分，加上独立的随机部分：

$$\epsilon_{t-1} = \underbrace{\sqrt{\frac{1-\bar{\alpha}_{t-1}-\sigma_t^2}{1-\bar{\alpha}_t}} \cdot \epsilon_t}_{\text{确定性部分（沿 } \epsilon_t \text{ 方向）}} + \underbrace{\frac{\sigma_t}{\sqrt{1-\bar{\alpha}_{t-1}}} \cdot z'}_{\text{随机部分}}$$

其中 $z' \sim \mathcal{N}(0, I)$ 与 $\epsilon_t$ 独立。

**验证方差：**

$$\text{Var}(\epsilon_{t-1}) = \frac{1-\bar{\alpha}_{t-1}-\sigma_t^2}{1-\bar{\alpha}_t} \cdot I + \frac{\sigma_t^2}{1-\bar{\alpha}_{t-1}} \cdot I$$

等等，这个分解不太对。让我们换一个更直接的方法。

### 2.5 直接构造法（DDIM 论文的方法）

**更清晰的推导方式：** 直接定义 $x_{t-1}$ 为 $x_0$、$x_t$ 和随机噪声的确定性函数。

从 $x_t$ 中，我们可以估计 $x_0$：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

然后定义 $x_{t-1}$ 为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \tilde{\epsilon}$$

其中 $\tilde{\epsilon}$ 是"指向 $x_t$ 方向"的噪声分量加上额外的随机噪声。

**具体地，** 我们将 $\tilde{\epsilon}$ 分解为：

$$\tilde{\epsilon} = \underbrace{\sqrt{\frac{1-\bar{\alpha}_{t-1}-\sigma_t^2}{1-\bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t)}_{\text{沿预测噪声方向的确定性分量}} + \underbrace{\sigma_t \cdot z}_{\text{额外随机噪声}}, \quad z \sim \mathcal{N}(0, I)$$

**为什么这样分解？**

- $\epsilon_\theta(x_t, t)$ 是神经网络预测的噪声方向，它包含了从 $x_t$ 中提取的关于噪声的信息
- 第一项沿着这个方向，但缩放系数使得 $x_{t-1}$ 的"确定性部分"的方差恰好是 $1-\bar{\alpha}_{t-1}-\sigma_t^2$
- 第二项是额外的随机性，方差为 $\sigma_t^2$
- 两部分加起来，总方差为 $1-\bar{\alpha}_{t-1}$，恰好等于 $q(x_{t-1}|x_0)$ 的方差

**验证（代入展开）：**

$$\begin{aligned}
x_{t-1} &= \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}} \cdot \sqrt{1-\bar{\alpha}_t} + \sigma_t \cdot z \\
&= \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot z
\end{aligned}$$

等等，让我更仔细地推导。实际上 DDIM 论文的做法是：

### 2.6 严格推导（按 DDIM 论文 Appendix C）

**Step 1：** 定义非马尔可夫前向过程

对于 $t \geq 2$，定义：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_\sigma(x_t, x_0, t), \sigma_t^2 I\right)$$

其中 $\sigma_t > 0$ 是可调参数。

**Step 2：** 求解均值 $\tilde{\mu}_\sigma$

我们需要 $x_{t-1} \sim q_\sigma(x_{t-1}|x_t, x_0)$ 的边缘分布等于 $q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$。

由 $q_\sigma$ 的定义，$x_{t-1} = \tilde{\mu}_\sigma + \sigma_t \epsilon'$，其中 $\epsilon' \sim \mathcal{N}(0, I)$。

同时，$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

**关键约束：** $x_{t-1}$ 的条件分布（给定 $x_0$）必须是 $\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$。

由于 $x_{t-1} = \tilde{\mu}_\sigma(x_t, x_0, t) + \sigma_t \epsilon'$，且 $x_t$ 本身是 $x_0$ 和 $\epsilon$ 的函数，我们需要选择 $\tilde{\mu}_\sigma$ 使得：

$$\tilde{\mu}_\sigma(x_t, x_0, t) + \sigma_t \epsilon' \sim \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$$

**DDIM 论文的解：**

$$\tilde{\mu}_\sigma(x_t, x_0, t) = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}$$

**验证：** 令 $\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}$，则：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon + \sigma_t \epsilon'$$

其中 $\epsilon$ 和 $\epsilon'$ 是独立的标准高斯。

$x_{t-1}$ 的均值：$\sqrt{\bar{\alpha}_{t-1}} x_0$ ✓

$x_{t-1}$ 的方差：$(1-\bar{\alpha}_{t-1}-\sigma_t^2 + \sigma_t^2)I = (1-\bar{\alpha}_{t-1})I$ ✓

**Step 3：** 用 $\epsilon_\theta$ 替换 $\epsilon$

在实际采样时，我们不知道真实的 $\epsilon$，用神经网络预测 $\epsilon_\theta(x_t, t)$ 代替。同时用 $\hat{x}_0$ 代替 $x_0$：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

代入 $\tilde{\mu}_\sigma$：

$$\tilde{\mu}_\sigma = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t)$$

**最终得到 DDIM 的一般采样公式：**

$$\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot z}$$

其中 $z \sim \mathcal{N}(0, I)$，$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$。

---

## 三、DDIM 的统一采样公式

### 3.1 σ_t 的选择与 η 参数化

上面的公式中，$\sigma_t$ 是一个自由参数。DDIM 论文引入 $\eta$ 参数来控制 $\sigma_t$：

$$\sigma_t = \eta \cdot \tilde{\beta}_t^{1/2} = \eta \cdot \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}$$

**为什么选这个形式？** 因为当 $\eta=1$ 时，$\sigma_t = \sqrt{\tilde{\beta}_t}$，恰好恢复 DDPM 的反向过程方差。

**代入验证：** 当 $\eta=1$ 时：

$$\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

此时 $1-\bar{\alpha}_{t-1}-\sigma_t^2 = 1-\bar{\alpha}_{t-1} - \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}$

$= (1-\bar{\alpha}_{t-1})\left(1 - \frac{\beta_t}{1-\bar{\alpha}_t}\right) = (1-\bar{\alpha}_{t-1}) \cdot \frac{1-\bar{\alpha}_t - \beta_t}{1-\bar{\alpha}_t}$

$= (1-\bar{\alpha}_{t-1}) \cdot \frac{\bar{\alpha}_t - \beta_t}{1-\bar{\alpha}_t} = (1-\bar{\alpha}_{t-1}) \cdot \frac{\alpha_t(1-\alpha_t^{-1}\bar{\alpha}_t)}{1-\bar{\alpha}_t}$

实际上更简单的验证方式是：将 DDIM 公式展开，与 DDPM 的 $\tilde{\mu}_t$ 公式对比，可以证明 $\eta=1$ 时两者完全一致。

### 3.2 统一公式的三种特例

$$\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\hat{x}_0(x_t, t)}_{\text{预测的 } x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot z}$$

| η 值 | σ_t | 公式简化 | 对应模型 |
|------|-----|----------|----------|
| $\eta=1$ | $\sqrt{\tilde{\beta}_t}$ | 恢复 DDPM 的随机采样 | **DDPM** |
| $\eta=0$ | $0$ | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta$ | **DDIM（纯 ODE）** |
| $0<\eta<1$ | 介于两者之间 | 部分随机 | **插值模式** |

### 3.3 DDIM（η=0）公式的另一种等价写法

当 $\eta=0$ 时，$\sigma_t=0$，公式变为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

将 $\hat{x}_0$ 展开：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

整理：

$$x_{t-1} = \sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} x_t + \left(\sqrt{1-\bar{\alpha}_{t-1}} - \sqrt{\frac{\bar{\alpha}_{t-1}(1-\bar{\alpha}_t)}{\bar{\alpha}_t}}\right) \epsilon_\theta$$

这个形式更清楚地展示了：**$x_{t-1}$ 是 $x_t$ 和 $\epsilon_\theta$ 的确定性函数，没有任何随机性。**

---

## 四、η 参数的核心作用

### 4.1 η 的不同取值

| $\eta$ 值 | 含义 | σ_t 值 | 对应模型 |
|-----------|------|--------|----------|
| $\eta = 1$ | 完全随机 | $\sqrt{\tilde{\beta}_t}$ | **DDPM**（恢复原版） |
| $\eta = 0$ | **完全确定** | $0$ | **DDIM**（纯 ODE） |
| $0 < \eta < 1$ | 半随机半确定 | 介于两者之间 | 中间状态 |

### 4.2 为什么 η=0 时是 ODE？

当 $\eta = 0$ 时：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t, t) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**没有随机项 $z$！** 这意味着：
- 给定 $x_T$，输出 $x_0$ 是**唯一确定**的
- 可以用任意步数（甚至 1 步！）完成采样
- 这是一个**常微分方程（ODE）**的离散化

更具体地说，当步长趋近于 0 时，DDIM 的更新规则收敛到一个 ODE：

$$\mathrm{d}x = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]\mathrm{d}t$$

这就是 Score SDE 论文中"概率流 ODE"的离散形式。

### 4.3 为什么 η=1 时恢复 DDPM？

当 $\eta=1$ 时，$\sigma_t = \sqrt{\tilde{\beta}_t}$，DDIM 公式变为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\tilde{\beta}_t} \cdot \epsilon_\theta + \sqrt{\tilde{\beta}_t} \cdot z$$

可以验证（通过代数展开），这等价于 DDPM 的采样公式：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right) + \sqrt{\tilde{\beta}_t} \cdot z$$

### 4.4 η 在 0 和 1 之间的效果

| η | 效果 | 适用场景 |
|---|------|----------|
| 0 | 完全确定性，可复现，可跳步 | 需要精确控制、图像编辑 |
| 0.2-0.5 | 轻微随机性，保持大部分确定性 | 需要少量多样性 |
| 0.5-0.8 | 中等随机性 | 平衡质量和多样性 |
| 1.0 | 完全随机，不可跳步 | 需要最大多样性 |

### 4.5 直觉理解

```
DDPM (η=1):                    DDIM (η=0):
x_T ──→ x_{T-1} ──→ ... ──→ x_0    x_T ──→ x_{T-1} ──→ ... ──→ x_0
 ↑                              ↑
 │每步加新噪声 z                 │无新噪声
 │必须走完1000步                │可以跳步！
 │每次结果不同                  │每次结果相同
 │像"醉汉走路"                  │像"沿着固定轨道滑行"
```

---

## 五、边缘分布一致性证明

这是 DDIM 的理论基石。我们需要证明：**非马尔可夫前向过程的边缘分布 $q_\sigma(x_t|x_0)$ 与 DDPM 的 $q(x_t|x_0)$ 完全相同。**

### 5.1 证明思路

**要证明：** 对于所有 $t$ 和所有 $\sigma > 0$，$q_\sigma(x_t|x_0) = q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$。

**方法：** 数学归纳法。

### 5.2 归纳基础

**$t = T$ 时：** $q_\sigma(x_T|x_0) = q(x_T|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_T}x_0, (1-\bar{\alpha}_T)I)$。

因为 $\bar{\alpha}_T \approx 0$（经过 1000 步累积后趋近于 0），所以 $x_T \approx \mathcal{N}(0, I)$，与 $\sigma$ 无关。✓

### 5.3 归纳步骤

假设 $q_\sigma(x_t|x_0) = q(x_t|x_0)$，证明 $q_\sigma(x_{t-1}|x_0) = q(x_{t-1}|x_0)$。

由全概率公式：

$$q_\sigma(x_{t-1}|x_0) = \int q_\sigma(x_{t-1}|x_t, x_0) \cdot q_\sigma(x_t|x_0) \, \mathrm{d}x_t$$

由归纳假设，$q_\sigma(x_t|x_0) = q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$。

由 DDIM 的定义：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_\sigma(x_t, x_0, t), \sigma_t^2 I)$$

其中 $\tilde{\mu}_\sigma = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}$。

**计算 $x_{t-1}$ 的分布：**

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon + \sigma_t \cdot \epsilon'$$

其中 $\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}} \sim \mathcal{N}(0, I)$，$\epsilon' \sim \mathcal{N}(0, I)$，两者独立。

**均值：**

$$\mathbb{E}[x_{t-1}] = \sqrt{\bar{\alpha}_{t-1}}x_0 + 0 + 0 = \sqrt{\bar{\alpha}_{t-1}}x_0 \quad \checkmark$$

**方差：**

$$\text{Var}(x_{t-1}) = (1-\bar{\alpha}_{t-1}-\sigma_t^2)I + \sigma_t^2 I = (1-\bar{\alpha}_{t-1})I \quad \checkmark$$

因此 $q_\sigma(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I) = q(x_{t-1}|x_0)$。✓

### 5.4 证明的意义

**这个证明告诉我们：**

1. **无论 $\sigma_t$（或 $\eta$）取什么值，边缘分布都不变**
2. **训练目标只依赖边缘分布，所以 DDIM 可以直接复用 DDPM 的训练权重**
3. **$\sigma_t$ 只影响采样过程，不影响训练过程**

---

## 六、为什么可以复用 DDPM 的训练权重

### 6.1 训练目标只依赖边缘分布

DDPM 的简化损失函数：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

**关键观察：** 这个损失函数只涉及 $q(x_t|x_0)$（边缘分布），不涉及 $q(x_t|x_{t-1})$ 或 $q(x_{t-1}|x_t)$（转移分布）。

由于 DDIM 保证了 $q_\sigma(x_t|x_0) = q(x_t|x_0)$，所以：

$$\mathcal{L}_{\text{DDIM}} = \mathcal{L}_{\text{DDPM}}$$

**训练目标完全相同！** 因此 DDIM 可以直接使用 DDPM 训练好的 $\epsilon_\theta$，无需重新训练。

### 6.2 更深层的理解

```
训练阶段：只关心 "给定 x₀，x_t 长什么样"
          → 只依赖边缘分布 q(x_t|x₀)
          → DDPM 和 DDIM 的训练数据完全相同

采样阶段：关心 "如何从 x_t 回到 x_{t-1}"
          → 依赖反向转移分布 p(x_{t-1}|x_t)
          → DDPM：随机转移（SDE）
          → DDIM：确定性转移（ODE）
          → 但两者看到的训练数据是一样的！
```

---

## 七、与 DDPM 的详细对比

### 7.1 公式对比

| 项目 | DDPM | DDIM ($\eta=0$) |
|------|------|----------------|
| **更新公式** | $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta) + \sqrt{\tilde{\beta}_t} \cdot z$ | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta$ |
| **随机项** | ✅ 有 $\sqrt{\tilde{\beta}_t} \cdot z$ | ❌ 无 |
| **数学形式** | SDE | **ODE** |
| **训练目标** | $\|\epsilon - \epsilon_\theta\|^2$ | **同左**（复用 DDPM 训练好的模型！） |
| **需要重新训练？** | — | ❌ 不需要！直接用 DDPM 权重 |

### 7.2 性能对比

| 维度 | DDPM | DDIM |
|------|------|------|
| 采样步数 | 1000 步 | **10-50 步** |
| 生成质量 | 高 | 几乎相同（步数≥50时） |
| 输出多样性 | 高（随机性） | 低（确定性） |
| 可控性 | 低 | **高**（可复现） |
| 速度 | ~30秒/张 | **~1秒/张** |
| 可插值 | 困难 | **容易**（潜空间插值） |

### 7.3 变量对比表

| 变量 | DDPM 中的含义 | DDIM 中的变化 |
|------|--------------|---------------|
| $\beta_t$ | 噪声调度（超参数） | **复用 DDPM 的值** |
| $\bar{\alpha}_t$ | 累积保留比例 | **同左** |
| $\epsilon_\theta$ | 噪声预测器 | **同左（直接复用！）** |
| $\sigma_t$ | 反向过程的噪声标准差 $\sqrt{\tilde{\beta}_t}$ | **由 $\eta$ 控制：$\sigma_t = \eta\sqrt{\tilde{\beta}_t}$** |
| $\eta$ | 不存在 | **新增：随机性控制参数** |
| $\hat{x}_0$ | 不显式使用 | **新增：预测的原始数据** |
| $z$ | 每步都需要的随机噪声 | **$\eta=0$ 时消失** |
| $\tilde{\beta}_t$ | $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ | **同左，用于计算 $\sigma_t$** |

---

## 八、DDIM 的完整算法

### 8.1 采样算法（$\eta=0$，确定性）

**算法：DDIM 确定性采样**

| 步骤 | 操作 |
|------|------|
| **输入** | 已训练的 $\epsilon_\theta$（来自 DDPM），子序列长度 $S$，总步数 $T$ |
| **初始化** | $x_{\tau_S} \sim \mathcal{N}(0, I)$，其中 $\tau_1 < \tau_2 < \cdots < \tau_S$ 是 $\{1,\ldots,T\}$ 的子序列 |
| **循环** | 对 $s = S, S-1, \ldots, 1$： |
| ① | 计算 $\hat{x}_0 = \frac{x_{\tau_s} - \sqrt{1-\bar{\alpha}_{\tau_s}}\epsilon_\theta(x_{\tau_s}, \tau_s)}{\sqrt{\bar{\alpha}_{\tau_s}}}$ |
| ② | $x_{\tau_{s-1}} = \sqrt{\bar{\alpha}_{\tau_{s-1}}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{\tau_{s-1}}} \epsilon_\theta(x_{\tau_s}, \tau_s)$ |
| **输出** | $x_0$（生成的数据） |

> 🎯 **关键优势：** 子序列 $\{\tau_s\}$ 可以任意选取！例如从 1000 步中均匀选 50 步。

### 8.2 采样算法（一般 $\eta$）

**算法：DDIM 一般采样**

| 步骤 | 操作 |
|------|------|
| **输入** | 已训练的 $\epsilon_\theta$，子序列 $\{\tau_s\}$，随机性参数 $\eta \in [0, 1]$ |
| **初始化** | $x_{\tau_S} \sim \mathcal{N}(0, I)$ |
| **循环** | 对 $s = S, S-1, \ldots, 1$： |
| ① | 计算 $\hat{x}_0 = \frac{x_{\tau_s} - \sqrt{1-\bar{\alpha}_{\tau_s}}\epsilon_\theta(x_{\tau_s}, \tau_s)}{\sqrt{\bar{\alpha}_{\tau_s}}}$ |
| ② | 计算 $\sigma_{\tau_s} = \eta \sqrt{\frac{1-\bar{\alpha}_{\tau_{s-1}}}{1-\bar{\alpha}_{\tau_s}} \beta_{\tau_s}}$ |
| ③ | 如果 $s > 1$：采样 $z \sim \mathcal{N}(0, I)$；否则 $z = 0$ |
| ④ | $x_{\tau_{s-1}} = \sqrt{\bar{\alpha}_{\tau_{s-1}}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{\tau_{s-1}}-\sigma_{\tau_s}^2} \cdot \epsilon_\theta(x_{\tau_s}, \tau_s) + \sigma_{\tau_s} \cdot z$ |
| **输出** | $x_0$ |

### 8.3 跳步策略示例

```
DDPM:   必须走完所有 1000 步
        [999] → [998] → [997] → ... → [1] → [0]

DDIM:   可以自由跳跃
        [999] → [980] → [960] → ... → [20] → [0]     ← 选50步
        [999] → [900] → [800] → ... → [100] → [0]    ← 选10步
        [999] → [500] → [0]                             ← 选2步
```

### 8.4 子序列选择策略

| 策略 | 公式 | 特点 |
|------|------|------|
| **均匀** | $\tau_s = \lfloor sT/S \rfloor$ | 简单，通常效果不错 |
| **二次方** | $\tau_s = \lfloor (s/S)^2 \cdot T \rfloor$ | 在低噪声区域（接近 $x_0$）更密集 |
| **余弦** | $\tau_s = \lfloor \cos(\frac{(S-s)\pi}{2S}) \cdot T \rfloor$ | 在两端更密集 |

实践中，**二次方调度**通常效果最好，因为接近 $x_0$ 的步骤对最终质量影响更大，需要更精细的采样。

---

## 九、DDIM 与 Neural ODE 的关系

### 9.1 Neural ODE 回顾

Neural ODE（Chen et al., 2018）将神经网络看作 ODE 的向量场：

$$\frac{\mathrm{d}x}{\mathrm{d}t} = f_\theta(x, t)$$

特点：
- 输入 $x(0)$，通过 ODE 求解器积分到 $x(T)$
- 确定性：相同输入 → 相同输出
- 可以用自适应步长求解器

### 9.2 DDIM 作为 Neural ODE

当 $\eta=0$ 时，DDIM 的更新公式：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)$$

可以重写为：

$$x_{t-\Delta t} - x_t = \left(\sqrt{\bar{\alpha}_{t-\Delta t}} - \sqrt{\bar{\alpha}_t}\right) \hat{x}_0 + \left(\sqrt{1-\bar{\alpha}_{t-\Delta t}} - \sqrt{1-\bar{\alpha}_t}\right) \epsilon_\theta(x_t, t)$$

当 $\Delta t \to 0$ 时，这收敛到一个 ODE：

$$\frac{\mathrm{d}x}{\mathrm{d}t} = \frac{\mathrm{d}}{\mathrm{d}t}\left[\sqrt{\bar{\alpha}_t}\right] \hat{x}_0 + \frac{\mathrm{d}}{\mathrm{d}t}\left[\sqrt{1-\bar{\alpha}_t}\right] \epsilon_\theta(x, t)$$

这正是 Score SDE 论文中的**概率流 ODE**的离散化形式。

### 9.3 实际意义

| Neural ODE 的性质 | DDIM 中的体现 |
|-------------------|---------------|
| 确定性 | $\eta=0$ 时输出完全确定 |
| 可逆性 | 可以从 $x_0$ 编码回 $x_T$（反向积分 ODE） |
| 变分下界 | ODE 的对数似然可以用 Hutchinson 迹估计器计算 |
| 自适应步长 | 可以用高阶 ODE 求解器（如 Runge-Kutta）加速 |

---

## 十、DDIM 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **速度快** | 50 步 vs 1000 步，提速 20 倍 |
| **无需重新训练** | 直接使用 DDPM 的权重 |
| **确定性输出** | 同输入 → 同输出，可复现 |
| **可控多样性** | 通过调节 $\eta$ 在确定性和多样性间切换 |
| **理论优雅** | 揭示了扩散模型的非马尔可夫结构 |
| **可逆性** | ODE 形式支持从 $x_0$ 编码回 $x_T$ |
| **潜空间插值** | 可以在 $x_T$ 空间插值，生成平滑过渡 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **仍依赖 DDPM 的 $\beta_t$** | 噪声调度仍是预设的超参数 |
| **小步数时质量下降** | 跳太多步会导致生成质量下降（ODE 求解精度不够） |
| **公式仍较复杂** | 大量 $\alpha$, $\beta$, $\bar{\alpha}$ 变量 |
| **损失函数未改进** | 复用 DDPM 的噪声预测损失 |
| **确定性可能限制多样性** | $\eta=0$ 时无法生成多样化的样本 |

---

## 十一、实践代码示例

### 11.1 DDIM 采样器（PyTorch）

```python
import torch

class DDIMSampler:
    def __init__(self, model, T=1000, beta_schedule='linear'):
        self.model = model
        self.T = T

        if beta_schedule == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, T)
        elif beta_schedule == 'cosine':
            steps = torch.arange(T + 1)
            self.beta = torch.clip(1 - (steps[1:] / T) / ((steps[:-1] / T) + 0.008) ** 2 * torch.pi / 2, 0.0001, 0.9999)

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample(self, shape, eta=0.0, S=50):
        """
        DDIM 采样
        Args:
            shape: 生成数据的形状 (B, C, H, W)
            eta: 随机性控制参数，0=确定性，1=DDPM
            S: 采样步数
        """
        device = next(self.model.parameters()).device
        b = shape[0]

        alpha_bar = self.alpha_bar.to(device)
        beta = self.beta.to(device)
        alpha = self.alpha.to(device)

        step_size = self.T // S
        timesteps = list(range(0, self.T, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1]

            t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)

            eps_pred = self.model(x, t_batch)

            alpha_bar_cur = alpha_bar[t_cur]
            alpha_bar_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            x0_pred = (x - torch.sqrt(1 - alpha_bar_cur) * eps_pred) / torch.sqrt(alpha_bar_cur)

            sigma_eta = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_prev)
            ) if eta > 0 else 0.0

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_eta ** 2) * eps_pred

            noise = torch.randn_like(x) if eta > 0 and t_prev > 0 else 0.0

            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_eta * noise

        return x

    @torch.no_grad()
    def encode(self, x0, S=50):
        """
        DDIM 编码：从 x_0 编码到 x_T（ODE 的反向积分）
        """
        device = x0.device
        alpha_bar = self.alpha_bar.to(device)

        step_size = self.T // S
        timesteps = list(range(0, self.T, step_size))

        x = x0
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((x0.shape[0],), t_cur, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)

            alpha_bar_cur = alpha_bar[t_cur]
            alpha_bar_next = alpha_bar[t_next]

            x0_pred = (x - torch.sqrt(1 - alpha_bar_cur) * eps_pred) / torch.sqrt(alpha_bar_cur)

            x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1 - alpha_bar_next) * eps_pred

        return x
```

### 11.2 使用示例

```python
model = ...  # 已训练的 DDPM 噪声预测网络
sampler = DDIMSampler(model, T=1000)

x_gen = sampler.sample(shape=(4, 3, 64, 64), eta=0.0, S=50)

x_gen_diverse = sampler.sample(shape=(4, 3, 64, 64), eta=0.5, S=50)

z = sampler.encode(x_real, S=50)
x_recon = sampler.sample_from_noise(z, eta=0.0, S=50)
```

---

## 十二、DDPM → DDIM 的演进脉络

```
DDPM (2020):                          DDIM (2021):
┌──────────────────────┐              ┌──────────────────────┐
│ 数学形式: SDE         │              │ 数学形式: ODE (η=0)   │
│                      │      发现      │                      │
│ 采样: 1000步          │  ──→ ──→       │ 采样: 10-50步         │
│                      │   非马尔可夫   │                      │
│ 损失: ‖ε - εθ‖²      │   过程         │ 损失: ‖ε - εθ‖²       │
│                      │   存在!        │ (直接复用!)           │
│ 需要: 设计β_t        │              │ 需要: 设计β_t         │
│                      │              │ 新增: η控制参数       │
│ 随机性: 不可控        │              │ 随机性: η可调         │
│ 可逆: 否             │              │ 可逆: 是(ODE)        │
└──────────────────────┘              └──────────────────────┘
 核心贡献: 扩散模型可行               核心贡献: 去掉随机性+加速+可逆
```

### 关键洞察总结

1. **DDPM 的马尔可夫假设不是必须的**——非马尔可夫过程可以产生相同的边缘分布
2. **训练只依赖边缘分布**——所以 DDIM 可以复用 DDPM 的权重
3. **随机性是可选的**——通过 $\eta$ 参数可以精确控制
4. **确定性 = 可跳步 = 快速**——ODE 形式允许任意步数采样
5. **ODE = 可逆**——可以从数据编码回噪声，实现潜空间操作

> **下一章：Score SDE —— 如何将 DDPM/DDIM 统一到更一般的框架中？**
