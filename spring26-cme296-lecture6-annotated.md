# [Spring 2026] CME 296 — Lecture 6: Training Lifecycle

> **来源：** `spring26-cme296-lecture6.pdf`（共 139 页幻灯片）
> **课程：** Stanford CME 296: Diffusion & Large Vision Models
> **讲师：** Afshine Amidi & Shervine Amidi
> **注解版本：** 严格按 PDF 原文顺序，含详细数学推导与补充注解

---

## 目录

1. [上讲回顾与范围](#一上讲回顾与范围)
2. [训练损失参数化](#二训练损失参数化)
3. [预训练](#三预训练)
4. [后训练](#四后训练)
5. [微调（Tuning）](#五微调tuning)
6. [蒸馏（Distillation）](#六蒸馏distillation)

---

## 一、上讲回顾与范围

### 1.1 上讲回顾

- **U-Net 及变体**：2015-2023，卷积架构
- **DiT 及变体**：2022-2026，Transformer 架构

### 1.2 本讲主题

**训练生命周期**——从预训练到部署的完整流程。

### 1.3 范围

本讲聚焦于**图像生成模型**的训练生命周期，不包括：
- VAE 编码器/解码器
- 嵌入模型（CLIP）
- 多模态 LLM

### 1.4 生命周期概览

| 阶段 | 目标 |
|------|------|
| **1. 预训练（Pre-training）** | 🖼 学会生成图像 |
| **2. 后训练（Post-training）** | 😃 学会生成好图像 |
| **3. 微调（Tuning）** | ⚙ 学会为特定场景生成图像 |
| **4. 蒸馏（Distillation）** | ⚡ 学会快速生成图像 |

> 📝 **注解：** 训练生命周期是一个从"能生成"到"生成好"到"生成特定内容"到"快速生成"的渐进过程。每个阶段都有不同的目标和技术，但它们共享同一个基础模型。

---

## 二、训练损失参数化

### 2.1 三种范式的损失函数

| 范式 | 损失函数 |
|------|---------|
| **扩散（DDPM）** | $\|\epsilon - \epsilon_\theta(z_t, t)\|^2$ |
| **Score Matching** | $\|s_\theta(z_t, t) + \epsilon/\sigma_t\|^2$ |
| **Flow Matching** | $\|v_\theta(x_t, t) - v_t(x_t|x_1)\|^2$ |

### 2.2 统一目标

三种范式都可以统一为：

$$\mathcal{L} = \mathbb{E}_{t, \text{data}, \text{noise}}\left[\|\text{prediction}_\theta - \text{target}\|^2\right]$$

**目标：** 预测"中间"量——既不是纯噪声也不是纯干净图像，而是介于两者之间的量。

### 2.3 任务难度随 $t$ 的变化

| $t$ 值 | 输入 | 任务难度 |
|--------|------|---------|
| 接近 0（干净） | 几乎是原图 | 简单 |
| 中间 | 部分加噪 | 中等 |
| 接近 1（纯噪声） | 几乎是纯噪声 | 困难 |

> 📝 **注解：** 当输入几乎是纯噪声时，模型需要从"几乎什么都没有"中预测目标——这非常困难。当输入接近原图时，预测任务相对简单。这种难度不均匀性是训练效率的关键瓶颈。

### 2.4 时间步分布的改进

**目标：** 在训练时强调"困难任务"

**想法：** 使用不同的时间步分布

**Logit-Normal 分布：**

$$p(t) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \frac{1}{t(1-t)} \exp\left(-\frac{(\text{logit}(t) - \mu)^2}{2\sigma^2}\right)$$

其中 $\text{logit}(t) = \log\frac{t}{1-t}$。

> 📝 **注解：** Logit-Normal 分布允许我们控制训练时间步的分布——通过调整 $\mu$ 和 $\sigma$，可以让模型更多地在"困难"时间步上训练。Esser et al. (2024) 在 SD3 中发现，偏向中间时间步的分布效果最好。

### 2.5 分辨率相关的时间步调整

**问题：** 相同噪声水平在不同分辨率下的"感知噪声"不同

**在更高分辨率下：**
- 相同噪声水平导致更少的"感知噪声"
- **解决方案：** 偏移时间步

> 📝 **注解：** 高分辨率图像有更多像素，相同方差的高斯噪声在高分辨率下"看起来"更弱（因为每个像素的噪声相对于图像整体更小）。SD3 通过偏移时间步来补偿这种效应——在高分辨率下使用更大的时间步值。

### 2.6 表示对齐（REPA）

> *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think*, Yu et al., 2024.

**REPA = REPresentation Alignment**

**想法：** 利用大型预训练模型学到的表示

### 2.7 REPA 损失

$$\mathcal{L}_{\text{REPA}} = 1 - \text{cos\_sim}\left(h_\phi(f_\theta(x_t, t)),\, g(x_1)\right)$$

其中：
- $f_\theta(x_t, t)$：生成模型 Transformer 编码器的输出
- $h_\phi$：可训练的投影头
- $g(x_1)$：干净图像的预训练表示

### 2.8 REPA 结果

- **显著加速训练**
- 只在前几层使用 REPA 效果最好
- 更大的扩散模型从 REPA 中受益更多

> 📝 **注解：** REPA 的核心洞察是：扩散模型的中间表示应该与预训练模型的表示对齐。这为训练提供了一个额外的监督信号——不仅要求模型预测正确的噪声/速度，还要求其中间表示与"好的"表示相似。这类似于知识蒸馏，但蒸馏的是表示而非输出。

---

## 三、预训练

### 3.1 预训练目标

🖼 **学会生成图像**

### 3.2 数据混合

使用多种来源的数据（图像-文本对），不同来源的数据质量和数量不同。

### 3.3 课程学习（Curriculum Learning）

**简单 → 困难**

| 阶段 | 设置 |
|------|------|
| **简单** | 低分辨率、固定正方形、简单短提示 |
| **困难** | 高分辨率、可变宽高比、复杂精细提示 |

> 📝 **注解：** 课程学习的直觉：先让模型学会生成简单图像（低分辨率、简单提示），然后逐步增加难度。这与人类学习过程类似——先学基础，再学高级技巧。

### 3.4 处理不同分辨率

**方法：** Patch embedding 机制天然支持不同分辨率——不同分辨率的图像产生不同数量的 patch token，但 DiT 可以处理任意长度的 token 序列。

- 低分辨率：较少的 patch
- 高分辨率：较多的 patch

> 📝 **注解：** DiT 的一个优势是天然支持可变分辨率——与 U-Net 的固定尺寸卷积不同，Transformer 的注意力机制可以处理任意长度的序列。这使得训练时可以使用混合分辨率数据。

---

## 四、后训练

### 4.1 后训练目标

😃 **学会生成好图像**

### 4.2 持续训练与监督微调

| 方法 | 关注点 |
|------|--------|
| **持续训练（CT）** | 知识——更多数据、更高质量数据 |
| **监督微调（SFT）** | 行为——光照、美学、指令遵循 |

> 📝 **注解：** CT 继续用大规模数据训练，侧重知识积累；SFT 用精心挑选的高质量数据训练，侧重行为塑造。两者互补——CT 让模型"知道更多"，SFT 让模型"表现更好"。

### 4.3 偏好微调概述

**想法：** 将人类偏好信号纳入图像生成方式

### 4.4 偏好微调方法

#### 4.4.1 Reward Feedback Learning（ReFL）

> *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*, Xu et al., 2023.

用奖励模型的反馈来调整生成模型。

#### 4.4.2 Flow-Group Reward Policy Optimization（Flow-GRPO）

> *Flow-GRPO: Training Flow Matching Models via Online RL*, Liu et al., 2025.

将强化学习中的 GRPO 方法适配到 Flow Matching 框架。

#### 4.4.3 Diffusion-DPO

> *Diffusion Model Alignment Using Direct Preference Optimization*, Wallace et al., 2023.

将 DPO（Direct Preference Optimization）应用于扩散模型——直接从偏好对中学习，无需奖励模型。

> 📝 **注解：** 三种偏好微调方法代表了不同的思路：ReFL 使用显式奖励模型，Flow-GRPO 使用在线强化学习，Diffusion-DPO 直接从偏好数据学习。DPO 方法最简单，因为它不需要训练额外的奖励模型。

### 4.5 优化输入提示

**PE = Prompt Enhancement**

将简短的用户提示自动扩展为更详细的描述，以获得更好的生成效果。

> 📝 **注解：** Prompt Enhancement 是一种"软"优化——不修改模型权重，而是优化输入。这在实践中非常有效，因为用户通常提供的提示太简短，而详细的提示能引导模型生成更高质量的图像。

---

## 五、微调（Tuning）

### 5.1 微调目标

⚙ **学会为特定场景生成图像**

### 5.2 DreamBooth

> *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*, Ruiz et al., 2022.

**训练：** 将权重调优到与特定主体关联的稀有 token

- 输入：几张特定主体的照片
- 用稀有 token `[V]` 关联主体
- 防止过拟合

**推理：** 使用稀有 token 生成包含该主体的新图像

例如：`[V] teddy bear attending CME 296`

> 📝 **注解：** DreamBooth 的关键技巧是使用稀有 token（如"xxy1z"）来关联特定主体。这避免了与已有概念的冲突。防止过拟合的方法是在训练时加入"先验保留"损失——用原始模型生成同类的其他图像，确保模型不会忘记原始概念。

### 5.3 LoRA

> *LoRA: Low-Rank Adaptation of Large Language Models*, Hu et al., 2021.

**LoRA = Low Rank Adaptation**

**想法：** 用两个低秩矩阵的乘积近似权重更新

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

**讨论：** 只需训练一小部分参数，性能相近。

> 📝 **注解：** LoRA 的核心思想是：权重更新矩阵 $\Delta W$ 是低秩的——即微调只需要在低维子空间中进行。这使得微调的参数量从 $d^2$ 降低到 $2dr$（$r \ll d$），极大减少了训练成本。

### 5.4 DreamBooth + LoRA 的讨论

| | 优势 | 劣势 |
|---|---|---|
| **DreamBooth** | 💯 推理开销最小，🖼 参考图像保真度高 | 💸 训练耗时且昂贵，❌ 不可复用 |

**使用场景：** 如果需要生成大量特定主体的图像，DreamBooth 是很好的方法。

---

## 六、蒸馏（Distillation）

### 6.1 蒸馏目标

⚡ **学会快速生成图像**

### 6.2 问题陈述

**动机：** 实际应用需要：
- 交互式使用
- 高吞吐量
- 动画制作

**约束：** 普通用户通常计算/金钱/时间有限

### 6.3 蒸馏的思路

**权衡：** 质量 / 速度

**利用：** 更大、更慢模型的丰富输出

**教师 vs 学生**

**目标：** 更快、更便宜的推理

### 6.4 "传统"蒸馏

**教师**（大）→ **学生**（小）

学生模仿教师的输出分布。

### 6.5 图像生成的蒸馏设定

**教师** → **学生**（通常大小相同）

教师：多步推理（如 50 步）
学生：少步推理（理想情况 1 步）

### 6.6 初始尝试

直接让学生从噪声一步跳到干净图像 → **太难了！**

### 6.7 渐进蒸馏（Progressive Distillation）

> *Progressive Distillation for Fast Sampling of Diffusion Models*, Salimans et al., 2022.

**PD = Progressive Distillation**

**想法：** 让教师走几小步，学生直接拟合这段转移

**过程：**

1. 教师走 2 步，学生走 1 步（步数减半）
2. 学生变成新的教师
3. 重复，直到步数降到目标

**结果：** 在相同 NFE 下比 DDIM 质量更好

> 📝 **注解：** 渐进蒸馏的核心思想是"分而治之"——不是一步到位地让学生学会从噪声到图像的映射，而是逐步让学生学会更大的步长。每次蒸馏将步数减半，经过 $\log_2 T$ 轮后，可以将 50 步模型蒸馏为 1 步。

### 6.8 渐进蒸馏的问题

- 弯曲路径 = 固定约束
- 需要离散、固定的采样方式
- 需要 $\log(T)$ 轮蒸馏

### 6.9 InstaFlow

> *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*, Liu et al., 2023.

**想法：** 利用 Reflow 过程（Lecture 3）

**过程：**
1. 从预训练模型开始
2. 应用 Reflow（让路径更直）
3. 将 N 步模型蒸馏为单步模型

**蒸馏损失：**

$$\mathcal{L}_{\text{distill}} = d(x_{\text{student}}, x_{\text{teacher}})$$

**距离函数：**
- Stage 1（"热身"）：MSE
- Stage 2（"打磨"）：LPIPS

**问题：** 更多 Reflow vs 更多蒸馏之间的权衡？

**结果：** 第一次 Reflow 带来最大收益

> 📝 **注解：** InstaFlow 的关键洞察是：先 Reflow 让路径变直，再蒸馏就更容易——因为直线路径可以用更少的步数精确近似。这与 Lecture 3 中 Rectified Flow 的思想一致。

### 6.10 InstaFlow 的局限

**计算负担：**
- 需要教师先铺好整条路
- 学到的路径被跳过
- 是否是利用信息的最佳方式？

### 6.11 一致性模型（Consistency Models）

> *Consistency Models*, Song et al., 2023.

**CM = Consistency Models**

**想法：** 从确定性 ODE 路径上的任意点预测干净图像

**自一致性性质：** 对于 ODE 路径上的任意两点 $x_t, x_{t'}$（$t, t' > 0$）：

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t')$$

即模型在 ODE 路径上的任何点都应映射到同一个干净图像。

**训练技术：**
- **一致性训练（CT）**：从零开始训练
- **一致性蒸馏（CD）**：利用教师模型

> 📝 **注解：** 一致性模型的核心约束是"自一致性"——同一个 ODE 轨迹上的所有点都应该映射到同一个结果。这保证了模型可以从任何噪声水平直接预测干净图像，实现一步生成。

### 6.12 一致性训练（CT）

**学生**与**自身**的一致性比较——相邻时间步的预测应该一致。

### 6.13 一致性蒸馏（CD）

**教师**（走一小步）与**学生**（直接预测）的一致性比较。

### 6.14 表示坍塌？

**不会！** 得益于边界条件 + stop-gradient 机制。

> 📝 **注解：** 一致性模型需要防止"表示坍塌"——如果所有输入都映射到同一个输出，模型就失去了意义。边界条件（$f_\theta(x_0, 0) = x_0$）和 stop-gradient 机制确保了模型不会坍塌。

### 6.15 一致性模型的结果与局限

**结果：** 蒸馏方法优于从头训练

**局限：质量**
- 回归到均值：模糊、平滑的结果
- 损失惩罚了错过目标图像的精确像素，即使图像有效且逼真

### 6.16 分布匹配蒸馏（DMD）

> *One-step Diffusion with Distribution Matching Distillation*, Yin et al., 2023.

**DMD = Distribution Matching Distillation**

**想法：** 匹配教师和学生的输出分布（而非逐像素匹配）

**关键洞察：** "更多听教师的，更少听学生的"

**损失函数：**

$$\mathcal{L}_{\text{DMD}} = \text{KL}(p_\text{student} \| p_\text{teacher})$$

**梯度更新：** 使用分数函数的差来近似 KL 梯度

> 📝 **注解：** DMD 与一致性模型的关键区别：一致性模型映射到**样本**（逐像素匹配），DMD 映射到**分布**（分布级匹配）。分布匹配更宽容——它不要求学生精确复制教师的每个像素，只要求整体分布相似。

### 6.17 DMD 的局限

**计算噩梦：**
- 蒸馏循环内有迷你扩散训练循环
- 两个模型计算大致相同的东西
- 如何用"硬"信号引导学生？

### 6.18 对抗扩散蒸馏（ADD）

> *Adversarial Diffusion Distillation*, Sauer et al., 2023.

**ADD = Adversarial Diffusion Distillation**

**想法：** 引入 GAN 的判别器

**结构：** 学生 + 教师建议 + 判别器（检测真/假）

**总损失：**

$$\mathcal{L}_{\text{ADD}} = \mathcal{L}_{\text{distillation}} + \mathcal{L}_{\text{adversarial}}$$

**判别器激励：** 区分真实图像和生成图像

**学生激励：** 欺骗判别器 + 接近教师输出

> 📝 **注解：** ADD 结合了蒸馏损失（保持与教师的一致性）和对抗损失（保持生成图像的逼真度）。判别器提供了"硬"信号——它直接告诉学生哪些图像看起来假，而不是像 DMD 那样提供"软"的分布匹配信号。

### 6.19 ADD 的局限

判别器在像素空间操作 → 需要在像素空间和潜在空间之间转换

### 6.20 潜在对抗扩散蒸馏（LADD）

> *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation*, Sauer et al., 2024.

**LADD = Latent Adversarial Diffusion Distillation**

**想法：** 将判别器损失重新构建在潜在空间中

**关键：** 教师模型本身充当特征提取器——判别器利用教师的中间表示来判断真/假

> 📝 **注解：** LADD 的创新在于：不需要额外的判别器网络——教师模型（扩散模型）本身就提供了丰富的特征表示。判别器只需要在教师的特征空间中区分真实和生成样本，这避免了像素空间的转换开销。

### 6.21 蒸馏方法总结

**方法论：**
- 教师/学生组合很强大
- 复杂度 vs 质量的权衡

**反复出现的要素：**
- 回归损失：MSE / LPIPS
- 分布/对抗压力
- 潜在空间训练

### 6.22 蒸馏方法的演进思路

```
渐进蒸馏 → 弯曲路径，log(T) 轮
    ↓ 让路径更直
InstaFlow → Reflow + 蒸馏
    ↓ 不需要教师铺路
一致性模型 → 自一致性约束
    ↓ 分布级匹配
DMD → 匹配输出分布
    ↓ 加入对抗信号
ADD → 蒸馏 + 对抗
    ↓ 在潜在空间操作
LADD → 潜在空间对抗蒸馏
```

> 📝 **注解：** 蒸馏方法的演进遵循一个清晰的逻辑：从"逐步逼近"到"直接映射"，从"逐像素匹配"到"分布匹配"，从"像素空间"到"潜在空间"。每一步都解决了前一步的某个关键局限，同时引入新的权衡。

---

## 附录：关键概念速查表

| 概念 | 定义/公式 | 注解 |
|------|---------|------|
| **Logit-Normal 分布** | 偏向中间时间步的 $t$ 分布 | 强调困难任务 |
| **REPA** | 表示对齐损失 | 利用预训练表示加速训练 |
| **课程学习** | 简单→困难的训练策略 | 先低分辨率后高分辨率 |
| **CT / SFT** | 持续训练 / 监督微调 | 知识 vs 行为 |
| **ReFL** | 奖励反馈学习 | 用奖励模型调整 |
| **Flow-GRPO** | 在线 RL 适配 Flow Matching | GRPO for Flow |
| **Diffusion-DPO** | 直接偏好优化 | 无需奖励模型 |
| **DreamBooth** | 稀有 token 关联主体 | 主体驱动的微调 |
| **LoRA** | $W' = W + BA$，低秩适应 | 大幅减少微调参数 |
| **渐进蒸馏** | 教师走 2 步，学生走 1 步 | 步数减半 |
| **InstaFlow** | Reflow + 蒸馏 | 先直化再蒸馏 |
| **一致性模型** | $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ | 自一致性约束 |
| **DMD** | 分布匹配蒸馏 | KL 散度匹配 |
| **ADD** | 蒸馏 + 对抗损失 | 判别器提供硬信号 |
| **LADD** | 潜在空间对抗蒸馏 | 教师充当特征提取器 |
