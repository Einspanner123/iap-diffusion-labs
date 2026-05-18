# [Spring 2026] CME 296 — Lecture 4: Multimodal Guided Generation

> **来源：** `spring26-cme296-lecture4.pdf`（共 148 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [上讲回顾](#一上讲回顾)
2. [Part 1：从像素空间到潜在空间](#二part-1从像素空间到潜在空间)
   - 2.1 像素空间的局限
   - 2.2 Autoencoder
   - 2.3 Variational Autoencoder（VAE）
   - 2.4 改进 VAE：感知损失与对抗损失
   - 2.5 潜在扩散模型（LDM）
3. [Part 2：文本表示](#三part-2文本表示)
   - 3.1 Tokenization
   - 3.2 Transformer 架构
4. [Part 3：图像表示](#四part-3图像表示)
   - 4.1 Vision Transformer（ViT）
5. [Part 4：对比学习与 CLIP](#五part-4对比学习与-clip)
6. [Part 5：引导生成](#六part-5引导生成)
   - 6.1 Classifier Guidance
   - 6.2 Classifier-Free Guidance

---

## 一、上讲回顾

### 1.1 三种生成范式

| | Lecture 1 | Lecture 2 | Lecture 3 |
|---|---|---|---|
| **方法** | 扩散（DDPM） | Score Matching（SDE） | Flow Matching |
| **预测目标** | 噪声 $\epsilon$ | Score $s$ | 速度 $v$ |
| **推理** | 逐步去噪 | 反向 SDE / PF-ODE | ODE 求解 |

### 1.2 统一视角

三种范式都可以用 SDE/ODE 框架统一。前向过程将数据扰动为噪声，反向过程将噪声变换回数据。

> 📝 **注解：** Lecture 4 将视角从"如何生成"转向"如何引导生成"——即如何让模型根据条件（如文本描述）生成特定图像。这需要三个组件：图像的潜在表示、文本的语义表示、以及将两者联系起来的引导机制。

---

## 二、Part 1：从像素空间到潜在空间

### 2.1 像素空间的局限

**朴素方法：** 在像素空间中表示和生成一切。

**三大问题：**

1. **高维度**：一张 $256 \times 256 \times 3$ 的图像有 196,608 维
2. **冗余信息**：相邻像素高度相关
3. **表示无意义**：在像素空间中移动一小步，图像就变成乱码

> 📝 **注解：** 像素空间的问题本质上是"维度灾难"——在高维空间中，大部分体积都是"无用"的（即不对应有意义的图像）。我们需要一个更低维、更紧凑、更有意义的表示空间。

### 2.2 理想空间的愿望清单

| 要求 | 含义 |
|------|------|
| **可处理的维度** | 维度足够低，计算可行 |
| **紧凑表示** | 去除冗余信息 |
| **有意义的表示** | 在空间中移动对应语义上有意义的变换 |

### 2.3 术语区分

| | 语义相似性 | 感知相似性 |
|---|---|---|
| **关注** | 结构、全局几何 | 局部、纹理 |
| **频率** | "低"频 | "高"频 |

> 📝 **注解：** 语义相似性关注"这是什么"（如两只不同姿态的猫），感知相似性关注"看起来怎样"（如纹理、颜色）。理想的表示空间应该同时捕捉这两种相似性。

### 2.4 尝试 1：Autoencoder

> *Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction*, Masci et al., 2011.

**结构：** 编码器 → 瓶颈层（潜在空间）→ 解码器

**目标：** 通过"代理任务"（重建）学习潜在表示

**编码器：** 通过卷积进行下采样

**解码器：** 通过反卷积进行上采样

**损失函数：** $\mathcal{L}_{\text{AE}} = \|x - \hat{x}\|^2$（输入与重建的像素距离）

**检查清单：**

| | 状态 |
|---|---|
| 可处理的维度 | ✅ |
| 紧凑表示 | ✅ |
| 有意义的表示 | ❌ |

> 📝 **注解：** Autoencoder 的问题在于潜在空间没有结构——编码器可以将数据映射到潜在空间中的任意位置，解码器也能还原。但潜在空间中"空白"区域解码出来可能是乱码。我们需要对潜在空间施加约束。

### 2.5 卷积快速回顾

**卷积操作：** 用滤波器（filter）在输入上滑动，提取局部特征（如边缘、纹理）

**步长（Stride）：** 滤波器每次移动的距离

**池化（Pooling）：** 下采样操作（最大池化 / 平均池化）

**转置卷积：** 上采样操作，相当于卷积的"反向"

### 2.6 尝试 2：Variational Autoencoder（VAE）

> *Auto-Encoding Variational Bayes*, Kingma et al., 2013.

**目标：** 在潜在空间上施加"结构"约束

**修改的编码器：** 输出均值 $\mu$ 和方差 $\sigma^2$（而非直接输出潜在向量）

**修改的解码器：** 假设常数方差（简化）

**重参数化技巧：** $z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$

### 2.7 VAE 损失函数推导

**Step 1：** 复用 Lecture 1 的 ELBO 技巧

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))$$

**Step 2：** 展开下界各项

**VAE 损失：**

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{D_{\text{KL}}(q(z|x) \| \mathcal{N}(0, I))}_{\text{KL 正则化}}$$

> 📝 **注解：** VAE 损失是重建损失和 KL 正则化的权衡。KL 项迫使编码器的输出接近标准正态分布，这给潜在空间施加了结构——使得潜在空间中的任意位置都能解码出有意义的图像。

### 2.8 检查清单

| | 状态 |
|---|---|
| 可处理的维度 | ✅ |
| 紧凑表示 | ✅ |
| 有意义的表示 | ✅ |
| 真实的表示 | ❌ |

> 📝 **注解：** VAE 的潜在空间有结构了，但重建的图像往往**模糊**——因为 VAE 的损失函数（MSE + KL）倾向于产生"平均"的输出，缺乏细节。

### 2.9 尝试 3：改进 VAE

**问题：** VAE 产生模糊图像

**原因分析：**
- **重建损失**（MSE）过高 → 产生模糊输出
- **KL 正则化**过高 → "后验坍塌"（posterior collapse），编码器忽略输入

### 2.10 感知损失（Perceptual Loss）

> *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, Zhang et al., 2018.

**想法：** 迫使模型关注人眼敏感的形状（边缘、轮廓），并具有一定平移不变性

**方法：** LPIPS（Learned Perceptual Image Patch Similarity）——用预训练网络的中间特征来衡量图像相似度

**权重：** 如果感知损失权重过高，可能产生"棋盘格伪影"（checkerboard artifacts）

> 📝 **注解：** LPIPS 不是在像素空间比较图像，而是在特征空间比较——这更符合人类感知。棋盘格伪影来自转置卷积的重叠不均匀，当感知损失过强时会放大这些伪影。

### 2.11 对抗损失（Adversarial Loss）

**想法：** 通过判别器（Discriminator）迫使解码器产生逼真的图像

**结构：** 解码器（生成器）+ 判别器

**权重：** 如果对抗损失过高，可能导致"模式坍塌"（mode collapse）——生成器忽略潜在编码，只产生少数几种图像

### 2.12 改进 VAE 损失总结

$$\mathcal{L}_{\text{VAE+}} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{重建}} + \underbrace{\mathcal{L}_{\text{perceptual}}}_{\text{感知}} + \underbrace{\mathcal{L}_{\text{adversarial}}}_{\text{对抗}} + \underbrace{\mathcal{L}_{\text{KL}}}_{\text{KL 正则化}}$$

> 📝 **注解：** 改进的 VAE 通过三种损失的组合来缓解模糊问题：MSE 保证基本重建，LPIPS 保证感知质量，对抗损失保证逼真度，KL 正则化保证潜在空间结构。Stable Diffusion 使用的 VAE 就是这种改进版本。

### 2.13 潜在扩散模型（Latent Diffusion Model, LDM）

> *High-Resolution Image Synthesis with Latent Diffusion Models*, Rombach et al., 2021.

**训练：**

1. 训练 VAE
2. 冻结 VAE 编码器，在 VAE 潜在空间中训练图像生成模型（扩散/Score/Flow Matching）

**推理：**

1. 从噪声开始在潜在空间中进行扩散/Score/Flow Matching
2. 用 VAE 解码器将潜在表示解码回像素空间

### 2.14 VAE 在图像生成模型中的作用

- **编码器：** 充当"低通滤波器"——去除高频细节，保留语义信息
- **解码器：** 负责"纹理幻觉"（texture hallucination）——从高度压缩的低维向量恢复高维图像
- **解码器通常是编码器的 2 倍大**——因为从低维到高维的映射更复杂

> 📝 **注解：** LDM 的核心洞察是：扩散过程不需要在像素空间进行——在潜在空间中进行更高效。编码器压缩图像，去除冗余；解码器恢复细节，添加纹理。这种"分工"使得生成模型可以专注于语义层面的生成，而将细节恢复交给解码器。

---

## 三、Part 2：文本表示

### 3.1 语言模型的演进

```
1980s: RNN → 1997: LSTM → 2013: Word2vec → 2017: Transformers → 2020s: LLMs
```

### 3.2 Tokenization

将文本分割成 token 的不同策略：

| 方法 | 示例 |
|------|------|
| **字符级** | A, c, u, t, e, t, e, d, d, y, ... |
| **词级** | A, cute, teddy, bear, is, reading |
| **子词级** | A, cute, ted, ##dy, bear, is, read, ##ing |

> 📝 **注解：** 子词级 tokenization（如 BPE, WordPiece）是最常用的方法——它平衡了词汇表大小和覆盖率。罕见词被分解为更常见的子词，避免了 OOV（out-of-vocabulary）问题。

### 3.3 Transformer 架构

> *Attention Is All You Need*, Vaswani et al., 2017.

**核心概念：**

- **注意力机制**：Query、Key、Value
- **代理任务**：下一个 token 预测
- **弱归纳偏置** → 更强的泛化能力

### 3.4 注意力机制

**QKV 概念：**
- **Query（查询）**：我在找什么
- **Key（键）**：我有什么信息
- **Value（值）**：我提供什么信息

**高效矩阵计算：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.5 注意力位置

| 类型 | 位置 |
|------|------|
| **自注意力** | 编码器-编码器 / 解码器-解码器 |
| **交叉注意力** | 编码器-解码器 |

### 3.6 Transformer 各组件

**输入：** 文本 tokenized → 学习的嵌入 + 位置编码

**编码器：** 自注意力 + FFN + 归一化

**解码器：** 自注意力 + 交叉注意力 + FFN + 归一化

**输出：** 线性投影 + softmax → 词汇表上的概率分布

**典型嵌入位置：** 编码器最后一层隐藏状态

> 📝 **注解：** Transformer 的编码器输出是文本的语义表示——每个 token 对应一个向量，捕捉了上下文信息。这个表示将在后续被用于引导图像生成。

---

## 四、Part 3：图像表示

### 4.1 将注意力机制适配到图像

**关键洞察：** 图像只是数字矩阵——也可以用注意力机制处理！

### 4.2 Vision Transformer（ViT）

> *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, Dosovitskiy et al., 2020.

**ViT = Vision Transformer**

**流程：**

1. 将图像分割成 $P \times P$ 的 patch
2. 每个 patch 线性投影为 $D$ 维向量
3. 添加 [CLS] token
4. 添加位置嵌入
5. 送入 Transformer 编码器
6. 用 [CLS] token 的输出做分类

### 4.3 ViT 端到端示例

```
图像 → 分割成 patches → 线性投影 → [CLS] + 位置嵌入 → Transformer 编码器 → 分类
```

### 4.4 ViT 的讨论

**局限：**
- 监督 ViT 效果好但需要标签
- 适合窄类别级表示，需要预先指定

**其他方法：**
- 文本不需要标签（自监督任务如 next token prediction）
- 流行工作：DINO（自蒸馏，无需标签）

> 📝 **注解：** ViT 的问题是：如何让图像和文本的嵌入在同一个空间中可比较？这引出了对比学习。

---

## 五、Part 4：对比学习与 CLIP

### 5.1 对比学习的动机

**问题：** 如何学习通用的图像/文本关系？

**想法：** 对比学习——将相似项拉近，将不相似项推远

### 5.2 对比学习的直觉

- 相似的图像-文本对（如"teddy bear"图像和"teddy bear"文本）→ 高相似度 → **拉近**
- 不相似的图像-文本对（如"teddy bear"图像和"water polo ball"文本）→ 低相似度 → **推远**

### 5.3 如何找到共同空间？

**图像编码器：** ViT → [CLS] → 投影 → 向量 $u$

**文本编码器：** Transformer 解码器 → 投影 → 向量 $v$

### 5.4 损失公式推导

**相似度：** $s_{\text{img, text}} = u^T v$

**图像匹配文本的概率：**

$$p(\text{teddy bear} | \text{🖼}) = \frac{\exp(s_{\text{🖼, teddy bear}})}{\exp(s_{\text{🖼, teddy bear}}) + \exp(s_{\text{🖼, water polo ball}})}$$

**文本匹配图像的概率：** 类似定义

**每个图像的损失：**

$$\mathcal{L}_{\text{🖼}} = -\log p(\text{正确文本} | \text{🖼})$$

**总损失：**

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{🖼}} + \mathcal{L}_{\text{text}})$$

### 5.5 CLIP

> *Learning Transferable Visual Models From Natural Language Supervision*, Radford et al., 2021.

**CLIP = Contrastive Language-Image Pretraining**

**扩展到 batch 大小 $N$：**

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{🖼→text}} + \mathcal{L}_{\text{text→🖼}})$$

其中每个方向的损失是 batch 内所有正确配对的对数概率之和。

### 5.6 CLIP 特点

**训练：**
- 数据集：4 亿 (图像, 标题) 对
- 视觉编码器：ViT 或 CNN
- 文本编码器：Transformer
- 损失依赖"批内负样本"假设

**结果：**
- 在 ImageNet 上无监督训练达到 ~76%
- 虽然可能存在监督集泄漏，但仍然令人印象深刻

### 5.7 CLIP 的讨论

**计算挑战：**
- 需要大型相似度矩阵
- 全局归一化
- 内存密集

**其他方法：**
- 将 1↔N softmax 问题重新构建为 1↔1 sigmoid 问题
- 流行工作：SigLIP（Sigmoid Loss for Language Image Pre-Training）

> 📝 **注解：** CLIP 的核心贡献是：将图像和文本映射到同一个嵌入空间，使得"teddy bear"的图像向量和"teddy bear"的文本向量在空间中靠近。这个共享嵌入空间是条件图像生成的关键——文本条件可以通过 CLIP 文本编码器转化为向量，然后注入生成模型。

---

## 六、Part 5：引导生成

### 6.1 动机

如何让生成模型根据条件（如类别标签、文本描述）生成特定图像？

### 6.2 第一个想法：Classifier Guidance

> *Diffusion Models Beat GANs on Image Synthesis*, Dhariwal et al., 2021.

**想法：** 用分类器的知识来引导图像生成

**结构：** 生成模型权重 + 分类器权重

### 6.3 Classifier Guidance 的推导

**目标：** 计算条件反向转移 $p(z_{t-1}|z_t, y)$（给定类别 $y$）

**推导路径：**

1. 对 $p(z_{t-1}|z_t)$ 应用条件贝叶斯法则
2. 结果分布与无条件分布相同，但均值偏移

**关键推导步骤：**

**Detour 1：** 利用 Lecture 1 的 DDPM 生成过程推导

$$\log p(z_{t-1}|z_t, y) = \log p(z_{t-1}|z_t) + \log p(y|z_{t-1}) + \text{const}$$

**Detour 2：** 对 $\log p(y|z_{t-1})$ 做一阶 Taylor 近似

$$\log p(y|z_{t-1}) \approx \log p(y|z_t) + \nabla_{z_t}\log p(y|z_t) \cdot (z_{t-1} - z_t) + \text{const}$$

**最终结果：** 条件分布与无条件分布形式相同，但均值偏移：

$$\mu_{\text{conditioned}} = \mu_{\text{unconditioned}} + \Sigma_t \nabla_{z_t}\log p(y|z_t)$$

> 📝 **注解：** Classifier Guidance 的核心是：条件生成 = 无条件生成 + 分类器梯度的修正。分类器的梯度 $\nabla_{z_t}\log p(y|z_t)$ 告诉我们"如何修改 $z_t$ 使其更像类别 $y$"，这个修正被加到无条件生成的均值上。

### 6.4 Classifier Guidance 的训练

**生成模型：** 无需重新训练无条件生成模型——纯粹是采样技术

**分类器：** 需要在加噪图像上用交叉熵损失训练，预测正确标签

### 6.5 给条件信号更多权重

**经验观察：** 需要放大分类器梯度才能获得好的条件样本

$$\mu_{\text{guided}} = \mu_{\text{unconditioned}} + s \cdot \Sigma_t \nabla_{z_t}\log p(y|z_t)$$

其中 $s > 1$ 是引导尺度（guidance scale）。

> 📝 **注解：** 引导尺度 $s$ 控制了条件信号的强度。$s$ 越大，生成结果越符合条件，但多样性越低。这是质量-多样性权衡的关键旋钮。

### 6.6 Classifier Guidance 的局限

**新前提条件：**
- 分类器需要在标注数据上训练
- 需要在加噪输入上操作：不能使用现成的分类器
- 两个模型之间的数据分布需要对齐

**计算考虑：**
- 每步需要额外的分类器前向传播
- 新的梯度信号需要缩放调优
- Taylor 展开引入近似误差

### 6.7 改进目标：仅使用生成模型权重

**动机：**
- 加噪数据上的分类器在自然界中不存在
- 希望避免推理时的额外模型调用

> 📝 **注解：** Classifier Guidance 的根本问题是需要一个额外的分类器。能不能去掉分类器，只用生成模型本身来实现引导？这就是 Classifier-Free Guidance 的动机。

### 6.8 Classifier-Free Guidance

> *Classifier-Free Diffusion Guidance*, Ho et al., 2022.

**关键观察：** 条件和无条件信号定义了一个隐式分类器！

$$\nabla_{z_t}\log p(y|z_t) \propto \nabla_{z_t}\log p(z_t|y) - \nabla_{z_t}\log p(z_t)$$

即：分类器梯度 = 条件 score - 无条件 score

### 6.9 思维转变

| | Classifier-Based | Classifier-Free |
|---|---|---|
| **分类器** | 外部分类器 | 隐式分类器（条件-无条件差） |
| **需要额外模型** | 是 | 否 |

**想法：** 同时计算条件和无条件情况，推断出分类器信号。

### 6.10 Classifier-Free Guidance 的训练

```
1. 采样：
   - 干净图像 z ~ p_data
   - 时间步 t
   - 噪声 ε ~ N(0, I)
   - 条件 c（以概率 p_drop 丢弃，变为 ∅）

2. 构造加噪样本并预测：
   z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε
   用模型预测：条件时 ε_θ(z_t, t, c)，无条件时 ε_θ(z_t, t, ∅)

3. 计算损失并反向传播：
   L = ‖ε_θ(z_t, t, c_or_∅) - ε‖²
```

> 📝 **注解：** Classifier-Free Guidance 的训练关键技巧是**条件丢弃**（condition dropping）——以一定概率（如 10%）将条件替换为空标记 ∅。这样同一个模型既学会了条件生成，也学会了无条件生成。推理时只需调用同一个模型两次。

### 6.11 Classifier-Free Guidance 的采样

$$\hat{\epsilon}_\theta(z_t, t, c) = \underbrace{\epsilon_\theta(z_t, t, \emptyset)}_{\text{无条件预测}} + s \cdot \left(\underbrace{\epsilon_\theta(z_t, t, c)}_{\text{条件预测}} - \underbrace{\epsilon_\theta(z_t, t, \emptyset)}_{\text{无条件预测}}\right)$$

- **无条件样本**：$\epsilon_\theta(z_t, t, \emptyset)$
- **条件样本**：$\epsilon_\theta(z_t, t, c)$
- **引导样本**：两者的加权差

> 📝 **注解：** 当 $s = 1$ 时，引导样本就是条件样本。当 $s > 1$ 时，条件信号被放大——生成结果更符合条件，但多样性降低。实践中 $s$ 通常在 3-10 之间。

### 6.12 实际考虑

**建模：**
- 条件的灵活性——CLIP 嵌入方便表示条件信号
- 最佳结果通常在 $s > 1$ 且条件丢弃概率约 10% 时获得

**计算：**
- 不再需要专门的分类器
- 但每步仍需要两次模型调用（条件 + 无条件）

> 📝 **注解：** Classifier-Free Guidance 是目前最主流的引导方法——Stable Diffusion、DALL-E、Imagen 等都使用它。虽然每步需要两次前向传播，但省去了训练额外分类器的麻烦，且效果通常更好。

---

## 附录：关键概念速查表

| 概念 | 定义/公式 | 注解 |
|------|---------|------|
| **Autoencoder** | 编码器→瓶颈→解码器，MSE 重建 | 潜在空间无结构 |
| **VAE** | MSE + KL 正则化 | 潜在空间有结构但图像模糊 |
| **LPIPS** | 感知损失，基于预训练网络特征 | 比像素 MSE 更符合人眼 |
| **LDM** | 在 VAE 潜在空间中做扩散 | Stable Diffusion 的核心 |
| **ViT** | 图像→patch→Transformer | 图像的 Transformer 表示 |
| **CLIP** | 对比学习，图像-文本共享嵌入 | 条件生成的文本编码器 |
| **Classifier Guidance** | $\mu + s \cdot \Sigma \nabla\log p(y\|z_t)$ | 需要额外分类器 |
| **Classifier-Free Guidance** | $\epsilon_\emptyset + s(\epsilon_c - \epsilon_\emptyset)$ | 不需要额外分类器 |
| **引导尺度 $s$** | 控制条件信号强度 | $s$↑ → 质量↑ 多样性↓ |
