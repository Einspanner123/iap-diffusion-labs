# 第10章 加速采样与最新研究进展

> 从 1000 步到 1 步，从像素空间到潜空间，从无条件生成到可控生成——扩散模型在过去几年经历了爆发式的改进。这一章介绍最重要的前沿进展，让你了解这个领域的最新面貌。

---

## 10.1 一致性模型（Consistency Models）

### 10.1.1 动机

DDPM 需要 1000 步采样，DDIM 可以减少到 20-50 步，但能否做到**1 步生成**？

一致性模型（Song et al., 2023）给出了肯定的答案。

### 10.1.2 核心思想

**一致性函数** $f_\theta(x, t)$ 满足**自一致性性质**：

$$f_\theta(x, t) = f_\theta(x', t'), \quad \forall t, t' \in (0, T]$$

也就是说：**同一条 ODE 轨迹上的任意两个点，通过一致性函数映射后，都应该得到相同的 $x_0$。**

```
ODE 轨迹上的点：
  x_T ──→ x_{t₃} ──→ x_{t₂} ──→ x_{t₁} ──→ x₀

一致性函数的映射：
  f(x_T, T) = f(x_{t₃}, t₃) = f(x_{t₂}, t₂) = f(x_{t₁}, t₁) = x₀
```

### 10.1.3 训练方法

**一致性蒸馏（Consistency Distillation）**：

利用预训练的扩散模型作为"教师"，训练一致性模型作为"学生"：

$$\mathcal{L}_{\text{CD}} = \mathbb{E}\left[\|f_\theta(x_{t+\Delta t}, t+\Delta t) - f_{\theta^-}(\hat{x}_t^\phi, t)\|^2\right]$$

其中：
- $\hat{x}_t^\phi$ 是用教师模型 $\phi$ 从 $x_{t+\Delta t}$ 一步去噪的结果
- $\theta^-$ 是 $\theta$ 的指数移动平均（EMA），保证训练稳定

**一致性训练（Consistency Training）**：

不需要教师模型，直接从数据训练：

$$\mathcal{L}_{\text{CT}} = \mathbb{E}\left[\|f_\theta(x_{t+\Delta t}, t+\Delta t) - f_{\theta^-}(x_t, t)\|^2\right]$$

### 10.1.4 采样

训练完成后，采样极其简单：

$$x_0 = f_\theta(x_T, T), \quad x_T \sim \mathcal{N}(0, I)$$

**只需 1 步！** 如果需要更高质量，可以用多步采样（交替去噪和加噪）。

---

## 10.2 蒸馏方法

### 10.2.1 渐进蒸馏（Progressive Distillation）

Salimans & Ho (2022) 提出：将 1000 步的模型蒸馏为 500 步，再蒸馏为 250 步，依此类推，最终得到 4 步甚至 1 步的模型。

**方法**：
1. 用教师模型走 2 步，得到 $x_{t-2}$
2. 训练学生模型从 $x_t$ 一步预测 $x_{t-2}$
3. 学生变成新的教师，重复

### 10.2.2 引导蒸馏（Guided Distillation）

Meng et al. (2023) 将无分类器引导（Classifier-free Guidance）也蒸馏到模型中，使得采样时不需要额外的引导计算。

### 10.2.3 TRACT

Berthelot et al. (2023) 提出分阶段蒸馏，每阶段只蒸馏一部分时间步，更稳定地收敛到少步模型。

---

## 10.3 潜空间扩散模型（Latent Diffusion Models）

### 10.3.1 动机

DDPM 在像素空间操作，一张 $512 \times 512$ 的图片是 $786432$ 维的向量！在高维空间做扩散，计算成本极高。

### 10.3.2 核心思想

**先降维，再扩散**：

1. 用自编码器将图片编码到低维潜空间：$z = E(x)$，$z \in \mathbb{R}^{d_z}$，$d_z \ll d_x$
2. 在潜空间做扩散：$z_0 \to z_T \to z_0$
3. 用解码器恢复图片：$\hat{x} = D(z_0)$

```
像素空间 x ∈ R^{786432}          潜空间 z ∈ R^{65536}
    │                                │
    │ 编码器 E                        │ 扩散模型
    ▼                                ▼
    z ────────────────────────────→ z_T → z_0
    │                                │
    │ 解码器 D                        │
    ▼                                │
    x̂ ←─────────────────────────────┘
```

### 10.3.3 Stable Diffusion

Rombach et al. (2022) 提出的 Latent Diffusion Model，即 **Stable Diffusion**：

- **编码器**：将 $512 \times 512 \times 3$ 的图片压缩到 $64 \times 64 \times 4$ 的潜空间
- **压缩比**：48 倍（$786432 / 16384$）
- **扩散模型**：在 $64 \times 64 \times 4$ 的潜空间运行 UNet
- **解码器**：从潜空间恢复图片

**为什么可以压缩？** 自然图像存在大量冗余——相邻像素高度相关，高频细节对语义影响不大。自编码器学会了丢弃冗余、保留语义的压缩方式。

### 10.3.4 自编码器的训练

Stable Diffusion 使用 **KL-正则化 VAE** 或 **VQ-VAE**：

- **KL 正则化**：约束潜变量接近标准高斯分布，便于扩散模型采样
- **VQ（向量量化）**：将潜空间离散化，避免潜空间"空洞"

---

## 10.4 引导生成（Classifier-free Guidance）

### 10.4.1 动机

无条件扩散模型可以生成各种图像，但无法控制生成内容。如何让模型根据文字描述（如"一只戴着墨镜的猫"）生成指定内容？

### 10.4.2 分类器引导（Classifier Guidance）

Dhariwal & Nichol (2021) 提出：在采样时，用分类器的梯度引导生成方向。

修改采样公式：

$$\tilde{\mu}_\theta = \mu_\theta + s \cdot \Sigma_\theta \nabla_{x_t} \log p_\phi(y|x_t)$$

其中 $p_\phi(y|x_t)$ 是分类器，$s$ 是引导强度。

**问题**：需要额外训练一个噪声感知的分类器，麻烦。

### 10.4.3 无分类器引导（Classifier-free Guidance）

Ho & Salimans (2022) 提出了更优雅的方案：**不需要分类器**。

**训练时**：以一定概率（如 10%）丢弃条件 $y$，训练模型同时学习条件生成 $\epsilon_\theta(x_t, t, y)$ 和无条件生成 $\epsilon_\theta(x_t, t, \varnothing)$。

**采样时**：用条件和无条件预测的差值作为引导：

$$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, \varnothing) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing))$$

其中 $s$ 是引导强度。

**直觉**：
- $\epsilon_\theta(x_t, t, y)$：知道条件 $y$ 时的去噪方向
- $\epsilon_\theta(x_t, t, \varnothing)$：不知道条件时的去噪方向
- 两者的差值：**条件 $y$ 提供的额外信息**
- 乘以 $s > 1$：放大条件的影响

**引导强度的效果**：

| $s$ | 效果 |
|-----|------|
| $s = 1$ | 标准条件生成 |
| $s = 3\text{-}7$ | 更强地遵循条件，图像更"锐利" |
| $s = 10\text{-}20$ | 过度引导，图像可能失真 |
| $s \to \infty$ | 退化为分类器引导 |

### 10.4.4 数学解释

从 Score 的角度看，无分类器引导等价于：

$$\hat{s}(x_t, t, y) = (1+s) \nabla_{x_t}\log p_t(x_t|y) - s \cdot \nabla_{x_t}\log p_t(x_t)$$

这相当于在条件 Score 和无条件 Score 之间插值，放大条件信息的影响。

---

## 10.5 DiT（Diffusion Transformers）

### 10.5.1 从 UNet 到 Transformer

DDPM 以来，扩散模型的骨干网络一直是 UNet。但 Peebles & Xie (2023) 提出了 **DiT（Diffusion Transformers）**——用 Transformer 替代 UNet。

### 10.5.2 DiT 的架构

```
输入：噪声潜变量 x_t（如 32×32×4）
  │
  ▼
Patchify：将 x_t 切成 patch 序列（如 2×2 patch → 16×16 = 256 个 token）
  │
  ▼
Transformer Blocks × N：
  每个块包含：
  - 自注意力层
  - 前馈网络
  - 自适应层归一化（AdaLN）：注入时间步 t 和条件 y
  │
  ▼
Unpatchify：将 token 序列恢复为潜变量形状
  │
  ▼
输出：预测的噪声 ε_θ 或速度 v_θ
```

### 10.5.3 自适应层归一化（AdaLN-Zero）

DiT 的关键创新是条件注入方式。标准的 Transformer 用固定的层归一化，DiT 用自适应层归一化：

$$\text{AdaLN}(h, c) = \gamma(c) \cdot \text{LayerNorm}(h) + \beta(c)$$

其中 $c$ 是条件（时间步 + 类别/文本），$\gamma$ 和 $\beta$ 是由条件生成的缩放和偏移参数。

**AdaLN-Zero**：初始化时将残差连接的最后一层设为 0，使得初始时 Transformer 块是恒等映射。这极大地加速了训练初期的收敛。

### 10.5.4 DiT 的优势

1. **可扩展性**：Transformer 的性能随模型大小和训练数据量平滑提升（Scaling Law）
2. **灵活性**：可以处理不同分辨率和长宽比的输入
3. **Sora 的基础**：OpenAI 的视频生成模型 Sora 就基于 DiT 架构

### 10.5.5 Scaling Law

DiT 论文发现，模型性能（FID）与计算量（GFLOPS）呈幂律关系：

$$\text{FID} \propto (\text{GFLOPS})^{-\alpha}$$

这意味着：**投入更多计算，就能得到更好的模型**——这是 Transformer 架构的核心优势。

---

## 10.6 其他重要进展

### 10.6.1 视频扩散模型

- **Video Diffusion Models**（Ho et al., 2022）：在时间和空间维度同时做扩散
- **Sora**（OpenAI, 2024）：基于 DiT 的视频生成模型，可生成最长 60 秒的高质量视频
- 核心挑战：时间一致性、运动物理、长序列计算

### 10.6.2 3D 生成

- **DreamFusion**（Poole et al., 2022）：用 2D 扩散模型引导 3D 生成（SDS 损失）
- **3D Gaussian Splatting**：用高斯点云表示 3D 场景，与扩散模型结合

### 10.6.3 离散扩散模型

- **D3PM**（Austin et al., 2021）：将扩散模型推广到离散数据（文本、分子）
- **VQ-Diffusion**：在离散潜空间做扩散

### 10.6.4 扩散模型用于科学

- **AlphaFold 3**：用扩散模型预测蛋白质-配体结构
- **DiffDock**：分子对接
- **GenCast**：天气预报

---

## 10.7 技术发展时间线

```
2020  DDPM（Ho et al.）—— 扩散模型的开山之作
  │
2021  DDIM（Song et al.）—— 确定性采样，20x 加速
  │   Score SDE（Song et al.）—— SDE 统一框架
  │   Classifier Guidance（Dhariwal & Nichol）—— 引导生成
  │
2022  Latent Diffusion / Stable Diffusion（Rombach et al.）—— 潜空间扩散
  │   Classifier-free Guidance（Ho & Salimans）—— 无分类器引导
  │   Progressive Distillation（Salimans & Ho）—— 蒸馏加速
  │   v-prediction（Salimans & Ho）—— 更稳定的参数化
  │
2023  DiT（Peebles & Xie）—— Transformer 架构
  │   Consistency Models（Song et al.）—— 1 步生成
  │   Flow Matching（Lipman et al.）—— 新训练范式
  │   Rectified Flow（Liu et al.）—— 直线流
  │   DALL·E 3（OpenAI）—— 文生图产品
  │
2024  Sora（OpenAI）—— 视频生成
  │   Stable Diffusion 3 —— Flow Matching + DiT
  │   Consistency Distillation —— 更好的蒸馏方法
  │   多模态扩散模型 —— 图文视频统一
```

---

## 10.8 本章小结

| 进展 | 解决的问题 | 核心方法 |
|------|-----------|---------|
| 一致性模型 | 采样太慢 | 自一致性约束，1 步生成 |
| 蒸馏方法 | 采样太慢 | 教师模型指导学生模型 |
| 潜空间扩散 | 计算成本高 | 先降维再扩散 |
| 无分类器引导 | 无法控制生成 | 条件与无条件预测的差值 |
| DiT | UNet 可扩展性差 | Transformer + AdaLN |

---

## 练习与思考

1. **无分类器引导**：设 $\epsilon_\theta(x_t, t, y) = (0.5, 0.3)^T$，$\epsilon_\theta(x_t, t, \varnothing) = (0.1, 0.1)^T$，$s = 7$。计算引导后的 $\hat{\epsilon}_\theta$。

2. **潜空间压缩**：一张 $512 \times 512 \times 3$ 的图片，编码到 $64 \times 64 \times 4$ 的潜空间。计算压缩比。

3. **一致性模型**：解释为什么自一致性约束 $f_\theta(x, t) = f_\theta(x', t')$ 能实现 1 步生成。

4. **思考题**：为什么 Transformer 比 UNet 更适合大规模扩展？（提示：考虑 Transformer 在 NLP 领域的 Scaling Law 经验）
