# 📚 MIT IAP 2026 · Lecture 4: Latent Spaces, Neural Network Architectures

## 潜空间与神经网络架构 —— 面向统计与概率论初学者

---

## 一、本讲两大主题

1. **Latent Spaces（潜空间）**：如何将高维图片压缩到低维空间，让扩散模型更高效
2. **Neural Network Architectures（神经网络架构）**：如何设计向量场的神经网络，特别是 Diffusion Transformer (DiT)

---

## 二、Section 6: 潜空间——为什么需要压缩？

### 2.1 高维度的困境

一张高分辨率图片的维度：

$$z \in \mathbb{R}^d, \qquad d = 3 \times 600 \times 1000 = 1{,}800{,}000$$

> **180万维！** 这带来三个问题：
> 1. **GPU 显存爆炸**：向量场 $u_t^\theta: \mathbb{R}^d \to \mathbb{R}^d$ 的输入输出都是180万维
> 2. **学习困难**：在如此高维空间学习分布极其困难
> 3. **冗余**：相邻像素高度相关（一片蓝天中，旁边的像素也大概率是蓝色）

**为什么扩散模型比普通监督学习更受影响？**
- 向量场 $u_t^\theta: \mathbb{R}^d \to \mathbb{R}^d$ 输出是高维的
- ODE 模拟需要**多次**调用网络（如50步），而不是只调用一次

### 2.2 解决方案：先压缩，再建模

$$\text{高维图片} \xrightarrow{\text{Encoder}} \text{低维潜变量} \xrightarrow{\text{Diffusion Model}} \text{生成潜变量} \xrightarrow{\text{Decoder}} \text{高维图片}$$

> 🎯 **核心思想：** 不直接在像素空间建模，而是先把图片"压缩"到一个小得多的空间（潜空间），在那里做扩散，最后再"解压"回来。

### 2.3 Autoencoder（自编码器）

Autoencoder = **Encoder**（编码器） + **Decoder**（解码器）

$$\text{Encoder}: \mathbb{R}^d \to \mathbb{R}^k, \qquad \text{Decoder}: \mathbb{R}^k \to \mathbb{R}^d$$

其中 $k \ll d$（**压缩！**）

**训练目标：** 输入 → 编码 → 解码 → 尽量还原输入

$$\mathcal{L}_{\text{recon}} = \|z - \text{Decoder}(\text{Encoder}(z))\|^2$$

### 2.4 普通 Autoencoder 的问题

编码后的潜空间分布可能**非常不规则**（比如所有点挤在一条窄线上）。

> 问题：扩散模型需要从简单分布（高斯）出发，如果潜空间分布太奇怪，学习将非常困难。

**目标：** 创建一个分布"漂亮"（接近标准高斯）的潜空间。

### 2.5 KL 散度——衡量两个分布的差异

$$D_{\text{KL}}(q(x) \| p(x)) = \int q(x) \log \frac{q(x)}{p(x)} \, \mathrm{d}x = \mathbb{E}_{X \sim q}\left[\log \frac{q(X)}{p(X)}\right]$$

**基本性质：**
- $D_{\text{KL}}(q \| p) \geq 0$（非负性）
- $D_{\text{KL}}(q \| p) = 0 \iff q = p$（相等时才为零）

**直觉：** KL 散度像一个"距离"（虽然不对称），衡量 $q$ 和 $p$ 有多不同。越大 = 差异越大。

### 2.6 两个高斯分布的 KL 散度

给定两个高斯分布：
$$q = \mathcal{N}(\mu_q, \sigma_q^2 I_d), \qquad p = \mathcal{N}(\mu_p, \sigma_p^2 I_d)$$

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\left(\mathcal{K}\!\left(\frac{\sigma_q^2}{\sigma_p^2}\right) + \frac{\|\mu_q - \mu_p\|^2}{\sigma_p^2}\right)$$

其中 $\mathcal{K}(\alpha) = \alpha - \log \alpha - 1$

**解读：**
- 第一项 $\mathcal{K}(\sigma_q^2/\sigma_p^2)$：**方差的差距**——两个分布的"宽度"有多不同
- 第二项 $\|\mu_q - \mu_p\|^2 / \sigma_p^2$：**均值的差距**——两个分布的"中心"有多远

> $\mathcal{K}(\alpha) = \alpha - \log \alpha - 1$ 在 $\alpha = 1$ 时取最小值 0，也就是当 $\sigma_q = \sigma_p$ 时方差项为零。

### 2.7 VAE（变分自编码器）——让潜空间"漂亮"

VAE 的关键改进：**在重构损失的基础上，加一个 KL 正则项**，强迫潜空间分布接近标准高斯 $\mathcal{N}(0, I_k)$。

**编码器输出**：不是一个固定向量，而是一个分布的**均值和方差**：

$$\text{Encoder}(z_i) \to (\mu_i, \log \sigma_i^2)$$

**重参数化采样**：

$$\hat{z}_i = \mu_i + \sigma_i \odot \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0, I_k)$$

> 这又是**重参数化技巧**（和 Flow Matching 中的一样）！

**$\beta$-VAE 训练算法（Algorithm 7）：**

```
输入: 数据 z ~ p_data, 编码器 (μ_φ, log σ²_φ), 解码器 μ_θ
对每个 mini-batch {z_1, ..., z_B}:
    1. 编码: μ_i = μ_φ(z_i), log σ²_i = log σ²_φ(z_i)
    2. 采样噪声: ε_i ~ N(0, I_k)
    3. 重参数化: ẑ_i = μ_i + σ_i ⊙ ε_i
    4. 解码: z̃_i = μ_θ(ẑ_i)
    5. 重构损失:
       L_recon = (1/B) Σ ‖z_i - z̃_i‖²
    6. KL 损失:
       L_KL = (1/B) Σ_i Σ_j (μ²_ij + σ²_ij - log σ²_ij - 1)
    7. 总损失: L = L_recon + β · L_KL
    8. 梯度下降更新 (φ, θ)
```

**两项损失的博弈：**

| 损失项 | 作用 | 比喻 |
|--------|------|------|
| $\mathcal{L}_{\text{recon}}$ | 重构尽量精确 | 压缩文件后还能还原 |
| $\beta \cdot \mathcal{L}_{\text{KL}}$ | 潜空间接近标准高斯 | 压缩后的空间要"整齐" |
| $\beta$ | 平衡两者 | $\beta$ 大 → 空间更整齐但还原差；$\beta$ 小 → 还原好但空间乱 |

### 2.8 Latent Diffusion Model (LDM) 完整流程

1. **数据**：收集训练图片 $x_1, \ldots, x_N$
2. **编码**：用 VAE 编码器将所有图片压缩为潜变量 $z_i = \text{Encoder}(x_i)$
3. **潜数据集**：得到低维数据集 $z_1, \ldots, z_N$（尺寸远小于原图）
4. **训练扩散模型**：在潜空间上用 Flow Matching 训练
5. **采样+解码**：生成潜变量 → Decoder 解码回图片

> 🎯 **本质就是换了一个数据集！** 所有之前学的 Flow Matching 公式完全不变，只是数据从"像素"变成了"潜变量"。

**实际压缩率惊人：**

| 模型 | 原始尺寸 | 潜空间尺寸 | 压缩倍数 |
|------|---------|-----------|---------|
| **Stable Diffusion** | [3, 256, 256] = 196K | [4, 32, 32] = 4K | **~48×** |
| **FLUX 2.0** | [3, 1024, 1024] = 3.1M | [32, 64, 64] = 131K | **~24×** |

---

## 三、Section 7: 神经网络架构——如何设计向量场网络

### 3.1 向量场网络的输入

$$u_t^\theta(x | y)$$

神经网络需要处理三种输入：

| 输入 | 含义 | 特点 |
|------|------|------|
| $t$ | 时间 | 1维标量 |
| $x$ | 潜图像 | 高维张量（如 $\mathbb{R}^{C \times H \times W}$） |
| $y$ | 文本 prompt | 可变长度的文本序列 |

### 3.2 时间编码——把1维变成高维

时间 $t$ 只有1维，但图像和 prompt 是高维的。为了让网络充分利用时间信息，使用**正弦位置编码**：

$$\text{TimeEmb}(t) = \frac{1}{\sqrt{d}}\left[\cos(2\pi w_1 t), \ldots, \cos(2\pi w_{d/2} t),\; \sin(2\pi w_1 t), \ldots, \sin(2\pi w_{d/2} t)\right]^T$$

其中频率 $w_i$ 按对数间距分布：

$$w_i = w_{\min}\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2 - 1}}, \qquad i = 1, \ldots, d/2$$

**性质：** $\|\text{TimeEmb}(t)\| = 1$（单位范数）

> 直觉：把一个标量"展开"成一个丰富的高维表示，低频成分捕捉粗略的时间位置，高频成分捕捉精细的时间变化。这和 Transformer 中的位置编码是同一个思想。

### 3.3 文本 Prompt 编码

使用预训练的语言模型将文本编码为向量序列：

$$\text{PromptEmbed}(y_{\text{raw}}) \in \mathbb{R}^{S \times k}$$

- $S$：文本序列长度（词数）
- $k$：每个词向量的维度
- 常用编码器：**CLIP**（粗粒度）、**T5-XXL**（序列级别）、LLM embeddings

### 3.4 图像 Patchify——把图像切成"碎片"

$$x \in \mathbb{R}^{C \times H \times W} \xrightarrow{\text{Patchify}} \tilde{x} \in \mathbb{R}^{L \times k}$$

- 将图像切成 $L$ 个不重叠的小方块（patches）
- 每个 patch 展平后线性投影到 $k$ 维
- 结果：一个**序列**，每个元素是一个 patch 的 $k$ 维向量

> 直觉：把图像从"像素网格"转变为"碎片序列"，这样就能用 Transformer 来处理了（Transformer 天生就处理序列）。

### 3.5 Diffusion Transformer (DiT)——核心架构

DiT 是专门为扩散模型设计的 Transformer 架构：

**整体流程：**

$$\underbrace{\tilde{t} = \text{TimeEmb}(t)}_{\mathbb{R}^k}, \quad \underbrace{\tilde{x}_0 = \text{PatchEmb}(x)}_{\mathbb{R}^{N \times k}}, \quad \underbrace{\tilde{y} = \text{PromptEmb}(y)}_{\mathbb{R}^{S \times k}}$$

$$\tilde{x}_{i+1} = \text{DiTBlock}(\tilde{x}_i, \tilde{t}, \tilde{y}) \in \mathbb{R}^{N \times k} \qquad (i = 0, \ldots, N_{\text{layers}}-1)$$

$$u = \text{Unpatchify}(\tilde{x}_{N_{\text{layers}}} W) \in \mathbb{R}^{C \times H \times W}$$

### 3.6 Attention 机制——DiT 的核心运算

**缩放点积注意力（Scaled Dot-Product Attention）：**

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

其中：
- $Q \in \mathbb{R}^{N \times d_h}$：查询（Queries）
- $K \in \mathbb{R}^{M \times d_h}$：键（Keys）
- $V \in \mathbb{R}^{M \times d_h}$：值（Values）
- $\sqrt{d_h}$：缩放因子，防止点积过大

> **直觉：** 每个 query 和所有 key 计算"相似度"（点积），用 softmax 归一化为权重，然后用这些权重对 values 做加权平均。就像"注意力"——每个位置决定它应该关注哪些其他位置的信息。

### 3.7 DiTBlock 的三种注意力

每个 DiTBlock 融合三种输入，用三种不同的注意力方式：

| 输入 | 注意力类型 | Q/K/V 设置 |
|------|-----------|-----------|
| **图像** | **Self-attention** | Q=图像, K=图像, V=图像 |
| **文本** | **Cross-attention** | Q=图像, K=文本, V=文本 |
| **时间** | **Adaptive Layer Norm** | 时间控制归一化的缩放和偏移 |

- **Self-attention**：图像 patch 之间互相"交流"（近处的树和远处的山要协调）
- **Cross-attention**：图像"查阅"文本信息（"这里应该画一只猫"）
- **Adaptive LayerNorm**：时间信息通过调节归一化参数来影响整个网络

---

## 四、Bonus: 大规模扩散模型实战

### 4.1 Case Study: Stable Diffusion 3

| 配置 | 值 |
|------|-----|
| 训练算法 | Flow Matching（直线调度） |
| 引导方式 | Classifier-free guidance ($w = 2.0 \sim 5.0$) |
| 空间 | 潜空间（预训练 VAE） |
| 网络架构 | **MM-DiT**（多模态 DiT） |
| 参数量 | **80亿** |
| 采样步数 | 50 步 |
| 文本编码器 | CLIP + T5-XXL |
| 数据集 | LAION |

### 4.2 Case Study: Meta MovieGen

| 配置 | 值 |
|------|-----|
| 训练算法 | Flow Matching（直线调度） |
| 引导方式 | Classifier-free guidance |
| 空间 | 潜空间（预训练 VAE，视频的时间维度使得压缩更关键） |
| 网络架构 | DiT（适配视频） |
| 参数量 | **300亿** |
| GPU | **6,144 张 H100** |

---

## 五、总结

### 完整的图像生成管道

```
训练阶段:
  原始图片 → [VAE Encoder] → 潜变量 → [Flow Matching 训练] → 学到 u_t^θ

生成阶段:
  随机噪声 → [ODE 模拟 (CFG)] → 潜变量 → [VAE Decoder] → 生成图片
```

### 核心公式汇总

| 组件 | 关键公式 | 作用 |
|------|---------|------|
| **VAE 重构损失** | $\mathcal{L}_{\text{recon}} = \|z - \text{Dec}(\text{Enc}(z))\|^2$ | 压缩后能还原 |
| **KL 正则** | $\mathcal{L}_{\text{KL}} = \sum(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$ | 潜空间接近高斯 |
| **Attention** | $\text{Attn}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_h}})V$ | 序列间信息交互 |
| **时间编码** | $\text{TimeEmb}(t) = [\cos(2\pi w_i t), \sin(2\pi w_i t)]$ | 标量→高维向量 |
| **Patchify** | $x \in \mathbb{R}^{C \times H \times W} \to \tilde{x} \in \mathbb{R}^{L \times k}$ | 图像→序列 |

> 🏆 **关键收获：** 现实中的大型图像/视频生成模型 = **VAE（压缩）** + **Flow Matching（训练）** + **CFG（引导）** + **DiT（架构）**。这四个组件组合在一起，就构成了 Stable Diffusion 3、FLUX、Meta MovieGen 等最先进系统的核心。
