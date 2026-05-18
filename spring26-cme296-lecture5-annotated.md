# [Spring 2026] CME 296 — Lecture 5: Image Generation Architectures

> **来源：** `spring26-cme296-lecture5.pdf`（共 171 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [问题陈述与目标](#一问题陈述与目标)
2. [U-Net](#二u-net)
3. [Diffusion Transformer（DiT）](#三diffusion-transformerdit)
4. [端到端示例](#四端到端示例)
5. [多模态 DiT（MM-DiT）](#五多模态-ditmm-dit)
6. [优化：位置嵌入](#六优化位置嵌入)

---

## 一、问题陈述与目标

### 1.1 上讲回顾

Lecture 4 建立了多模态引导生成的框架：VAE（潜在空间）+ 文本/图像表示 + CLIP + 引导机制。

### 1.2 本讲主题

**图像生成架构**——为生成模型选择合适的神经网络架构。

### 1.3 问题陈述

生成模型的输入：
- 加噪潜在表示（noisy latent）
- 噪声水平（timestep）
- 条件（condition，如文本描述）

输出：
- 预测的速度/噪声/score

### 1.4 补充说明

- 条件也可以是其他形式（如"让这只泰迪熊跳舞"）
- 输出也可以是噪声或 score（取决于生成范式）

### 1.5 目标

选择满足以下标准的图像生成架构：

- 理解全局结构
- 保留局部细节
- 响应外部信号（时间步、条件）
- 计算可扩展

> 📝 **注解：** 架构选择是生成模型的关键决策之一。U-Net 和 DiT 是两种主流选择——U-Net 依赖卷积的归纳偏置，DiT 依赖注意力的全局建模能力。本讲将详细介绍两者。

---

## 二、U-Net

### 2.1 卷积模型的直觉

**想法：** 通过"归纳偏置"模仿人类视觉

### 2.2 卷积操作与滤波器

**卷积滤波器：** 特征检测器，检测边缘、角点、纹理、模式、形状

**步长（Stride）：** 滤波器每次移动的距离

### 2.3 感受野

**感受野（Receptive Field）：** 激活映射能"看到"的区域

一个像素能看到的区域大小为 $\frac{F}{S}$，其中 $F$ 是滤波器大小，$S$ 是步长。

> 📝 **注解：** 感受野决定了每个输出像素能"看到"多大范围的输入。深层网络的感受野更大，能捕捉全局结构；浅层网络的感受野更小，能保留局部细节。

### 2.4 下采样：池化操作

**池化：** 每个通道的下采样操作，通常在卷积之后

- **最大池化**：取窗口内的最大值
- **平均池化**：取窗口内的平均值

### 2.5 上采样：转置卷积

**转置卷积：** 上采样操作，相当于"广播"

### 2.6 U-Net 架构

> *U-Net: Convolutional Networks for Biomedical Image Segmentation*, Ronneberger et al., 2015.

**概述：**
- 同时捕捉局部和全局特征
- 与反向扩散过程相关的归纳偏置
- 输入/输出维度相同

### 2.7 U-Net：下采样阶段

- 卷积和最大池化的混合
- 每步特征数量减半
- 提取相关特征
- 类似于"编码器"

### 2.8 U-Net：上采样阶段

- 上卷积和卷积的混合
- 每步特征数量加倍
- 最后一步 1×1 卷积得到输出
- 类似于"解码器"

### 2.9 U-Net：残差连接

**"复制并裁剪"连接：**
- 避免丢失局部信息
- 信息的"高速公路"

> 📝 **注解：** 残差连接是 U-Net 的核心创新——它将下采样阶段的特征直接传递给上采样阶段的对应层，使得解码器可以同时利用"压缩的语义信息"和"原始的局部细节"。这与 ResNet 的跳跃连接思想一致。

### 2.10 U-Net：添加条件信息

**问题：** 如何将时间步和类别标签信息注入 U-Net？

### 2.11 时间步的表示

**正弦位置编码**（类似时钟）：
- 不同频率的正弦/余弦函数
- 低频 = "慢"变化（小时），高频 = "快"变化（秒）

### 2.12 类别标签的表示

**预定义类别：** 单个嵌入向量

**自由文本：** 来自预训练 LLM 的丰富嵌入（多个 token 向量）

### 2.13 条件注入的三种方法

| 方法 | 描述 |
|------|------|
| **方法 1** | 添加到特征映射 |
| **方法 2** | 通过缩放/偏移调制特征映射 |
| **方法 3** | 条件与特征映射的交叉注意力 |

> 📝 **注解：** 方法 1 最简单但最不灵活；方法 2 是 AdaBN/AdaIN 的思路，在扩散模型中最常用；方法 3 提供了最细粒度的交互，但计算量最大。

### 2.14 U-Net 模型时间线

```
2015: 原始 U-Net
2020: 像素空间 U-Net + DDPM
2021: 潜在空间 U-Net + LDM（Stable Diffusion）
2023: 大型 U-Net + SDXL
```

---

## 三、Diffusion Transformer（DiT）

### 2.1 动机

**问题：** 需要在长距离上保留局部细节（如"泰迪熊的爪子"和"泰迪熊的脸"之间的关系）

**U-Net 的局限：** 卷积的感受野有限，需要很深的网络才能建立长距离依赖

### 3.2 注意力机制

> *Attention Is All You Need*, Vaswani et al., 2017.

**想法：** 移除归纳偏置

注意力机制允许每个位置直接"看到"所有其他位置——不受距离限制。

### 3.3 Transformer

**多头注意力层 + Transformer 架构**

### 3.4 ViT 用于视觉理解

> *An Image is Worth 16x16 Words*, Dosovitskiy et al., 2020.

**ViT = Vision Transformer**

### 3.5 DiT 用于视觉生成

> *Scalable Diffusion Models with Transformers*, Peebles et al., 2022.

**DiT = Diffusion Transformer**

### 3.6 DiT：输入的 tokenization

**加噪潜在表示 → Patchify → 输入 token**

将加噪的潜在表示分割成 patch，每个 patch 线性投影为一个 token。

### 3.7 DiT：条件嵌入

- **时间步嵌入**：正弦位置编码
- **标签嵌入**：类别标签的向量

### 3.8 DiT：用自适应 Layer Norm 注入条件

**Adaptive Layer Norm（adaLN）**

### 3.9 DiT：用交叉注意力注入条件

### 3.10 DiT：用上下文条件注入（In-context Conditioning）

将条件 token 直接拼接到 patch token 序列中。

### 3.11 DiT：条件注入方式对比

| 排名 | 方法 |
|------|------|
| 🥇 | Adaptive Layer Norm |
| 🥈 | 交叉注意力 |
| 🥉 | 上下文条件注入 |

> 📝 **注解：** Peebles et al. (2022) 的实验表明，adaLN 在 DiT 中效果最好。这可能是因为 adaLN 通过调制整个特征映射来注入条件，与扩散模型"全局调整噪声水平"的需求更匹配。

### 3.12 自适应 Layer Norm 的直觉

**patch 嵌入的表示：** 包含颜色、形状、纹理等信息

**早期阶段（高噪声）：** 需要关注全局结构 → 条件信号应强调"全局布局"

**后期阶段（低噪声）：** 需要关注局部细节 → 条件信号应强调"纹理细节"

> 📝 **注解：** 自适应 Layer Norm 的优势在于它可以根据条件（时间步+标签）动态调整每个 token 的表示。在生成早期，条件信号帮助模型关注"大局"；在生成后期，条件信号帮助模型关注"细节"。

### 3.13 自适应 Layer Norm 的步骤

**Step 1：** 根据条件确定调制强度

时间步 + 类别标签 → 投影 → 调制向量 $(\gamma, \beta, \alpha)$

**Step 2：** 用调制向量调制 token

$$\hat{x} = \underbrace{(1 + \gamma)}_{\text{Scale}} \odot \text{LayerNorm}(x) + \underbrace{\beta}_{\text{Shift}}$$

$$\text{output} = \underbrace{\alpha}_{\text{Gate}} \odot \hat{x}$$

> 📝 **注解：** adaLN-Zero 的关键设计是 Gate $\alpha$ 初始化为 0——这意味着训练开始时，DiT block 相当于恒等映射。这种初始化策略极大地加速了训练收敛，类似于 ResNet 的零初始化思想。

### 3.14 DiT：adaLN-Zero 条件注入

调制表示使用：
- **Scale**：$(1 + \gamma)$
- **Shift**：$\beta$
- **Gate**：$\alpha$

作为条件 $(c)$ 的函数。

### 3.15 DiT：输出重格式化

Token → Layer Norm → 线性投影 → Reshape → 预测的速度/噪声

### 3.16 扩展 DiT

**FLOPs** = 浮点运算数 = 前向传播中的操作数

- 参数量不足以量化模型复杂度
- 更小的 patch 大小对应更高的 FLOPs
- **扩展 DiT 改善结果！**

> 📝 **注解：** DiT 的核心优势是可扩展性——与 U-Net 不同，DiT 可以通过增加模型大小和训练数据来持续改善性能。Peebles et al. (2022) 展示了 FLOPs 与 FID 之间的清晰缩放规律。

---

## 四、端到端示例

### 4.1 完整流程

```
"teddy bear" → 采样噪声潜在 → Patchify → 嵌入条件 → DiT Block → 预测速度 → 更新潜在 → 解码
```

### 4.2 采样噪声潜在

在 VAE 学习的潜在空间中采样纯噪声。

### 4.3 Patchify 噪声潜在

将潜在表示分割成 patch → 每个 patch 做线性投影（patch embedding）

### 4.4 嵌入条件

时间步嵌入 + 标签嵌入 → 条件嵌入

### 4.5 DiT Block 内部流程

1. **位置嵌入**：添加到 patch 嵌入 → 位置感知 patch 嵌入
2. **条件处理**：条件嵌入 → MLP → Gate + Scale + Shift
3. **Patch 处理**：
   - LayerNorm → Scale & Shift → 自注意力 → Gate
   - LayerNorm → Scale & Shift → FFN → Gate
4. **输出**：上下文化的潜在状态矩阵

### 4.6 投影和重塑

上下文化潜在状态 → Layer Norm → 线性投影 → Reshape → 预测的速度

### 4.7 预测回顾

```
DiT 输入：类别标签 "teddy bear" + 时间步 + 加噪潜在
DiT 输出：预测的速度
```

### 4.8 推导结果潜在

用预测的速度更新潜在表示（Euler 步）。

### 4.9 迭代

重复 DiT 预测 + Euler 更新，直到获得最终潜在。

### 4.10 解码

用 VAE 解码器将最终潜在解码回像素空间 → 生成图像。

---

## 五、多模态 DiT（MM-DiT）

### 5.1 动机：回到直觉

**问题：** 当文本提示更微妙时（如"brown fluffy teddy bear surrounded by white walls"），adaLN 的调制对所有 patch 相同——无法区分"棕色区域"和"白色区域"。

**局限：** 所有 patch 潜在受相同"调制"。

### 5.2 缓解策略

1. **保持时间步嵌入的调制**（adaLN）
2. **对条件使用更复杂的交互方式**：
   - 交叉注意力
   - 联合注意力（Joint Attention）

### 5.3 联合注意力的直觉

将 patch 嵌入和条件嵌入放在一起做自注意力——让两种模态在注意力层中直接交互。

### 5.4 MM-DiT

> *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, Esser et al., 2024.

**MM-DiT = MultiModal-Diffusion Transformer**

**三种变体：**

| 变体 | 描述 |
|------|------|
| **Single-stream** | 所有模态平等对待 |
| **Double-stream** | 每个模态在自己的流中 |
| **Hybrid** | 包含 single-stream 和 double-stream 层 |

> 📝 **注解：** MM-DiT 是 Stable Diffusion 3 引入的架构。Single-stream 将图像和文本 token 混在一起做注意力；Double-stream 让图像和文本各有独立的注意力流，然后通过交叉注意力交互；Hybrid 结合两者。

### 5.5 "Double-stream" DiT 变体：SD3, Qwen-Image

图像和文本各有独立的注意力流，通过交叉注意力交互。

### 5.6 "Single-stream" DiT 变体：Z-Image

图像和文本 token 混合在一起做自注意力。

### 5.7 Hybrid 方法：FLUX.1 Kontext

包含 single-stream 和 double-stream 层。

### 5.8 时间线

```
2022: DiT
2023: "MM-DiT" coined
2024: Stable Diffusion 3 (Double-stream)
2025: Qwen-Image (Double-stream), FLUX.1 (Hybrid)
2026: Z-Image (Single-stream)
```

---

## 六、优化：位置嵌入

### 6.1 位置信息的必要性

**动机：** 直接连接"丢失"位置信息——注意力是排列不变的，无法区分不同位置的 token。

**希望：** 点积能传达 token 表示之间的距离信息。

### 6.2 绝对位置嵌入

**想法：** 将位置特定的嵌入添加到 token 向量

$$e = e_{\text{token}} + e_{\text{position}}$$

### 6.3 硬编码位置嵌入

**提议：** 正弦和余弦的混合

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**性质：** $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数

> 📝 **注解：** 正弦位置编码的关键性质是：任意位置 $pos+k$ 的编码可以表示为位置 $pos$ 编码的线性变换。这使得模型可以通过学习线性变换来捕捉相对位置关系。

### 6.4 硬编码位置嵌入的讨论

**优势：**
- 实现简单
- 作为基线效果不错
- 给每个位置直接的表示身份

**局限：**
- 位置信息没有注入到"正确"的位置（注意力层）
- 副作用：额外的交互项
- 位置嵌入也进入了 value 嵌入

### 6.5 使用绝对位置嵌入的模型

- 原始 Transformer（2017）
- 原始 ViT（2020）
- 原始 DiT（2022）

### 6.6 从绝对到相对位置信息

**改进：** 将位置信息移到直觉上需要的地方——注意力层。

### 6.7 默认选择：旋转位置嵌入（RoPE）

> *RoFormer: Enhanced Transformer with Rotary Position Embeddings*, Su et al., 2021.

**RoPE = Rotary Position Embeddings**

**想法：** 用旋转矩阵旋转 query 和 key 向量

**优势：** 相对距离被很好地捕捉——有效旋转角度只依赖于相对距离。

> 📝 **注解：** RoPE 的核心思想是：不把位置信息加到 token 嵌入上，而是在计算注意力时旋转 query 和 key。两个位置的点积只依赖于它们的相对距离，而不是绝对位置。这使得 RoPE 天然具有平移不变性。

### 6.8 推广到 2D

**文本：** 1D 位置 → RoPE 天然适用 ✅

**图像：** 2D 位置 → 如何推广？❓

### 6.9 第一个想法：Axial RoPE

将偶数索引用于一个轴，奇数索引用于另一个轴。

**问题：** 没有跨轴交互 + 分离信息是任意的

### 6.10 2D RoPE

**提议：** 将两个轴混合到同一个旋转矩阵中

**结果：** 避免轴伪影

> 📝 **注解：** 2D Mixed RoPE 将 x 和 y 坐标混合在同一个旋转中，而不是分别处理。这避免了 Axial RoPE 中"两个轴不交互"的问题，使得对角线方向的位置关系也能被正确捕捉。

### 6.11 缩放 RoPE

**问题：** 分辨率变化时，空间含义也变了

**解决方案：** 缩放到规范坐标

> 📝 **注解：** 当训练和推理的分辨率不同时，相同的 patch 位置可能对应不同的语义含义（如"中心"在不同分辨率下对应不同的像素位置）。通过将坐标归一化到 [-1, 1] 的规范空间，可以保持空间语义的一致性。

### 6.12 多模态 RoPE

**问题：** 如何协调 1D 文本和 2D 图像的位置信息？

**MSRoPE = Multimodal Scalable RoPE**

**优势：**
- 图像和文本可以共存而不互相干扰
- 文本功能上等价于 1D RoPE

> 📝 **注解：** MSRoPE 是 Qwen-Image 提出的方法——它为不同模态使用不同的位置编码维度（文本用 1D，图像用 2D），但在同一个注意力层中统一处理。这使得模型可以同时处理文本和图像的位置信息。

### 6.13 结论

**位置嵌入 = 开放问题**

- 有很多变体
- 尘埃尚未落定
- 存在权衡

---

## 附录：关键概念速查表

| 概念 | 定义/公式 | 注解 |
|------|---------|------|
| **U-Net** | 编码器-解码器 + 残差连接 | 卷积归纳偏置 |
| **DiT** | Transformer 用于扩散 | 可扩展，adaLN-Zero |
| **adaLN-Zero** | $(1+\gamma) \odot \text{LN}(x) + \beta$，Gate 初始化为 0 | DiT 最佳条件注入 |
| **MM-DiT** | 多模态 DiT | SD3 的架构 |
| **Single-stream** | 图像+文本混合注意力 | Z-Image |
| **Double-stream** | 图像+文本各有独立流 | SD3, Qwen-Image |
| **Hybrid** | 混合 single/double-stream | FLUX.1 |
| **RoPE** | 旋转 query/key 的位置编码 | 捕捉相对距离 |
| **2D Mixed RoPE** | 混合两个轴的旋转 | 避免轴伪影 |
| **MSRoPE** | 多模态可缩放 RoPE | 1D 文本 + 2D 图像 |
| **FLOPs** | 浮点运算数 | 衡量模型复杂度 |
