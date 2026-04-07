# 第二代：DDIM (Denoising Diffusion Implicit Models)

> **论文：** Denoising Diffusion Implicit Models (Song et al., 2021)
>
> **地位：** 发现 DDPM 可以改写为非马尔可夫过程，去掉随机项后变成 ODE，采样速度提升 20 倍

---

## 一、核心思想

### 1.1 从 DDPM 出发的问题

回顾 DDPM 的采样公式：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)}_{\text{确定性部分 } f(x_t, t)} + \underbrace{\sigma_t z}_{\text{随机部分}}, \quad z \sim \mathcal{N}(0, I)$$

**问题：**
- 每一步都有随机项 $\sigma_t z$ → 需要走完所有 1000 步才能保证质量
- 无法跳步 → 不能加速
- 输出不确定 → 相同输入产生不同结果

### 1.2 DDIM 的关键洞察

> **DDPM 的 SDE 不是唯一的反向路径！存在一族非马尔可夫的反向过程，它们都共享相同的边缘分布 $q(x_{t-1}|x_0)$。**

这意味着：**我们可以选择一个"更好"的反向路径——确定性、可跳跃的 ODE。**

---

## 二、从 DDPM 到 DDIM 的推导

### 2.1 回顾 DDPM 的前向过程

DDPM 的前向（加噪）过程：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)$$

重参数化：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 2.2 关键观察：非马尔可夫反向过程

DDPM 假设反向过程是**马尔可夫的**：

$$p_\theta(x_{T:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t) \quad \text{(DDPM: 只依赖前一步)}$$

DDIM 放宽这个假设，允许**非马尔可夫**的过程：

$$p_\theta(x_{0:T}) = q(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t, x_0) \quad \text{(DDIM: 可以依赖 x₀)}$$

**约束条件：** 只要保证边缘分布正确：
$$q(x_{t-1} | x_0) = \int q(\phi) p_\theta(x_{t-1} | x_t, x_0) q(x_t | x_0) d x_t$$

其中 $\phi$ 是某个辅助变量。

### 2.3 推导 DDIM 更新公式

#### Step 1：定义新的时间调度

引入一组新的参数 $\sigma_t$（注意：这里的 $\sigma_t$ 与 DDPM 中的含义不同）：

$$\sigma_t^2 + \tau_t^2 = 1$$

其中 $\tau_t$ 控制确定性部分的权重。

#### Step 2：构造一般形式的反向分布

$$p_\theta(x_{t-1} | x_t, x_0) = q(x_{t-1}^\sigma | x_t, x_0)$$

其中 $x_{t-1}^\sigma$ 定义为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \tilde{\epsilon}_{t-1}$$

这里 $\tilde{\epsilon}_{t-1}$ 是待确定的噪声方向。

#### Step 3：利用 $x_t$ 和 $x_0$ 的关系

已知 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

我们可以把 $\tilde{\epsilon}_{t-1}$ 表示为 $\epsilon$ 的函数。DDIM 选择如下形式：

$$\tilde{\epsilon}_{t-1} = \frac{\sqrt{1-\bar{\alpha}_{t-1}} - \sigma_t^2}{\sqrt{1-\bar{\alpha}_t}} \cdot \epsilon + \sigma_t \cdot z$$

其中 $z \sim \mathcal{N}(0, I)$。

#### Step 4：代入得到最终更新公式

经过代数运算（展开并整理），得到：

$$\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot z}$$

这个公式看起来很复杂，但可以简化为更直观的形式。

### 2.4 简化形式

令 $\eta = \sigma_t / \sqrt{1-\bar{\alpha}_t}$ 作为控制随机性的超参数，则上式等价于：

$$\boxed{
\begin{aligned}
x_{t-1} &= \sqrt{\bar{\alpha}_{t-1}} \underbrace{\hat{x}_0(x_t, t)}_{\text{预测的 } x_0} + \sqrt{1-\bar{\alpha}_{t-1} - \eta^2 \beta_t} \cdot \epsilon_\theta(x_t, t) + \eta \sqrt{\beta_t} \cdot z \\
\hat{x}_0(x_t, t) &= \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
\end{aligned}
}$$

| 参数 | 含义 |
|------|------|
| $\hat{x}_0(x_t, t)$ | 神经网络预测的原始数据（去噪后的结果） |
| $\eta$ | **随机性控制参数**（核心创新！） |
| $\beta_t$ | 来自 DDPM 的噪声调度 |

---

## 三、$\eta$ 参数的核心作用

### 3.1 $\eta$ 的不同取值

| $\eta$ 值 | 含义 | 对应模型 |
|-----------|------|----------|
| $\eta = 1$ | 完全随机 | **DDPM**（恢复原版） |
| $\eta = 0$ | **完全确定** | **DDIM**（纯 ODE） |
| $0 < \eta < 1$ | 半随机半确定 | 中间状态 |

### 3.2 为什么 $\eta=0$ 时是 ODE？

当 $\eta = 0$ 时：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t, t) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**没有随机项 $z$！** 这意味着：
- 给定 $x_T$，输出 $x_0$ 是**唯一确定**的
- 可以用任意步数（甚至 1 步！）完成采样
- 这是一个**常微分方程（ODE）**的离散化

### 3.3 直觉理解

```
DDPM (η=1):                    DDIM (η=0):
x_T ──→ x_{T-1} ──→ ... ──→ x_0    x_T ──→ x_{T-1} ──→ ... ──→ x_0
 ↑                              ↑
 │每步加新噪声 z                 │无新噪声
 │必须走完1000步                │可以跳步！
 │每次结果不同                  │每次结果相同
```

---

## 四、与 DDPM 的详细对比

### 4.1 公式对比

| 项目 | DDPM | DDIM ($\eta=0$) |
|------|------|----------------|
| **更新公式** | $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta) + \sigma_t z$ | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta$ |
| **随机项** | ✅ 有 $\sigma_t z$ | ❌ 无 |
| **数学形式** | SDE | **ODE** |
| **训练目标** | $\|\epsilon - \epsilon_\theta\|^2$ | **同左**（复用 DDPM 训练好的模型！） |
| **需要重新训练？** | — | ❌ 不需要！直接用 DDPM 权重 |

### 4.2 性能对比

| 维度 | DDPM | DDIM |
|------|------|------|
| 采样步数 | 1000 步 | **10-50 步** |
| 生成质量 | 高 | 几乎相同（步数≥50时） |
| 输出多样性 | 高（随机性） | 低（确定性） |
| 可控性 | 低 | **高**（可复现） |
| 速度 | ~30秒/张 | **~1秒/张** |

### 4.3 变量对比表

| 变量 | DDPM 中的含义 | DDIM 中的变化 |
|------|--------------|---------------|
| $\beta_t$ | 噪声调度（超参数） | **复用 DDPM 的值** |
| $\bar{\alpha}_t$ | 累积保留比例 | **同左** |
| $\epsilon_\theta$ | 噪声预测器 | **同左（直接复用！）** |
| $\sigma_t$ | 反向过程的噪声标准差 | **被 $\eta$ 取代** |
| $\eta$ | 不存在 | **新增：随机性控制参数** |
| $\hat{x}_0$ | 不显式使用 | **新增：预测的原始数据** |
| $z$ | 每步都需要的随机噪声 | **$\eta=0$ 时消失** |

---

## 五、DDIM 的完整算法

### 5.1 采样算法（$\eta=0$）

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

### 5.2 跳步策略示例

```
DDPM:   必须走完所有 1000 步
        [999] → [998] → [997] → ... → [1] → [0]

DDIM:   可以自由跳跃
        [999] → [980] → [960] → ... → [20] → [0]     ← 选50步
        [999] → [900] → [800] → ... → [100] → [0]    ← 选10步
        [999] → [500] → [0]                             ← 选2步
```

---

## 六、DDIM 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **速度快** | 50 步 vs 1000 步，提速 20 倍 |
| **无需重新训练** | 直接使用 DDPM 的权重 |
| **确定性输出** | 同输入 → 同输出，可复现 |
| **可控多样性** | 通过调节 $\eta$ 在确定性和多样性间切换 |
| **理论优雅** | 揭示了扩散模型的非马尔可夫结构 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **仍依赖 DDPM 的 $\beta_t$** | 噪声调度仍是预设的超参数 |
| **小步数时质量下降** | 跳太多步会导致生成质量下降 |
| **公式仍较复杂** | 大量 $\alpha$, $\beta$, $\bar{\alpha}$ 变量 |
| **损失函数未改进** | 复用 DDPM 的噪声预测损失 |

---

## 七、DDPM → DDIM 的演进脉络

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
└──────────────────────┘              └──────────────────────┘
 核心贡献: 扩散模型可行               核心贡献: 去掉随机性+加速
```

> **下一章：Score SDE —— 如何将 DDPM/DDIM 统一到更一般的框架中？**
