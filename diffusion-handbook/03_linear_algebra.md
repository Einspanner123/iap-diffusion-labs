# 第3章 线性代数基础：向量、矩阵与变换

> 一张 256×256 的彩色图片，本质上就是一个 196608 维的向量。线性代数就是处理这种高维数据的语言。在扩散模型中，噪声是向量的加法，协方差是矩阵，而整个扩散过程可以看作高维空间中的线性变换。

---

## 3.1 向量与向量空间

### 3.1.1 什么是向量？

在高中物理中，你学过向量是有方向和大小的量（如力、速度）。在线性代数中，向量更一般——它就是**一组有序的数**：

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{pmatrix} \in \mathbb{R}^d$$

**在扩散模型中**：一张 $H \times W$ 的灰度图片可以"拉直"成一个 $d = H \times W$ 维的向量。一张 $256 \times 256$ 的彩色图片（3 个颜色通道）就是 $d = 256 \times 256 \times 3 = 196608$ 维的向量！

### 3.1.2 向量的基本运算

**加法**：对应分量相加

$$\mathbf{x} + \mathbf{y} = \begin{pmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_d + y_d \end{pmatrix}$$

**扩散模型中的例子**：DDPM 的前向过程就是向量加法！

$$x_t = \underbrace{\sqrt{\bar{\alpha}_t} \cdot x_0}_{\text{信号部分}} + \underbrace{\sqrt{1 - \bar{\alpha}_t} \cdot \epsilon}_{\text{噪声部分}}$$

这里 $x_0$（原始图像）和 $\epsilon$（噪声）都是高维向量，$x_t$ 是它们的加权求和。

**数乘**：每个分量乘以标量

$$c \cdot \mathbf{x} = \begin{pmatrix} cx_1 \\ cx_2 \\ \vdots \\ cx_d \end{pmatrix}$$

在扩散模型中，$\sqrt{\bar{\alpha}_t}$ 就是对 $x_0$ 的缩放因子，$\sqrt{1 - \bar{\alpha}_t}$ 是对噪声 $\epsilon$ 的缩放因子。

### 3.1.3 内积与范数

**内积（点积）**：

$$\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{d} x_i y_i = \mathbf{x}^T \mathbf{y}$$

内积的几何意义：$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$，其中 $\theta$ 是两个向量的夹角。

- 内积为 0 → 两个向量垂直（正交）
- 内积为正 → 两个向量方向大致相同
- 内积为负 → 两个向量方向大致相反

**范数（模、长度）**：

$$\|\mathbf{x}\| = \sqrt{\mathbf{x} \cdot \mathbf{x}} = \sqrt{\sum_{i=1}^{d} x_i^2}$$

**扩散模型中的例子**：DDPM 的损失函数就是噪声预测误差的范数平方：

$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2 = \sum_{i=1}^{d}(\epsilon_i - \epsilon_{\theta,i})^2$$

---

## 3.2 矩阵与线性变换

### 3.2.1 矩阵：向量的变换规则

矩阵 $A \in \mathbb{R}^{m \times n}$ 可以看作一个"机器"：输入一个 $n$ 维向量，输出一个 $m$ 维向量。

$$\mathbf{y} = A\mathbf{x}$$

**矩阵乘法的规则**：$(A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij} x_j$

### 3.2.2 特殊矩阵

| 矩阵 | 定义 | 在扩散模型中的角色 |
|------|------|-------------------|
| 单位矩阵 $I$ | 对角线为 1，其余为 0 | 高斯分布的协方差矩阵 |
| 对角矩阵 $\Lambda$ | 只有对角线非零 | 各维度独立的高斯分布 |
| 对称矩阵 $A = A^T$ | $A_{ij} = A_{ji}$ | 协方差矩阵总是对称的 |
| 正交矩阵 $Q^TQ = I$ | 列向量互相正交且单位长度 | 保持距离的变换（旋转、反射） |

### 3.2.3 协方差矩阵——扩散模型的关键概念

给定一组向量 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N$，均值为 $\bar{\mathbf{x}}$，协方差矩阵定义为：

$$\Sigma = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

**直觉**：
- 对角线元素 $\Sigma_{ii}$：第 $i$ 个维度的方差（该维度的"分散程度"）
- 非对角线元素 $\Sigma_{ij}$：第 $i$ 和第 $j$ 个维度的协方差（两个维度的"关联程度"）

**在扩散模型中**：高斯分布 $\mathcal{N}(\mu, \Sigma)$ 完全由均值 $\mu$ 和协方差 $\Sigma$ 决定。DDPM 假设每一步的噪声协方差是对角矩阵 $\beta_t I$，这意味着**噪声在每个维度上是独立同分布的**。

### 3.2.4 矩阵的迹与行列式

**迹（Trace）**：对角线元素之和

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

迹有一个重要性质：$\text{tr}(AB) = \text{tr}(BA)$

**在扩散模型中**：计算高斯分布之间的 KL 散度时需要用到迹。

**行列式（Determinant）**：衡量矩阵对空间的"缩放"程度

- $|\det(A)| = 1$：保持体积不变（如旋转）
- $\det(A) = 0$：将空间压缩到更低维度（不可逆）
- $\det(A) > 0$：保持方向
- $\det(A) < 0$：翻转方向

**在扩散模型中**：概率密度变换公式（变量替换）需要行列式：

$$p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|$$

其中 $J_{f^{-1}}$ 是逆变换的雅可比矩阵。Flow Matching 和正则化流的核心就是利用这个公式。

---

## 3.3 特征值与特征向量

### 3.3.1 定义

对于方阵 $A$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得：

$$A\mathbf{v} = \lambda \mathbf{v}$$

那么 $\lambda$ 是 $A$ 的**特征值**，$\mathbf{v}$ 是对应的**特征向量**。

**直觉**：特征向量是矩阵作用下"方向不变"的向量，特征值是"伸缩比例"。

### 3.3.2 对称矩阵的特征分解

任何实对称矩阵 $A$ 都可以分解为：

$$A = Q \Lambda Q^T$$

其中 $Q$ 是正交矩阵（列向量是特征向量），$\Lambda$ 是对角矩阵（对角线是特征值）。

### 3.3.3 在扩散模型中的应用

**协方差矩阵的特征分解**：高斯分布 $\mathcal{N}(\mu, \Sigma)$ 中，$\Sigma$ 是对称正定矩阵，可以分解为 $\Sigma = Q \Lambda Q^T$。特征值表示数据在每个主方向上的"扩展程度"。

**主成分分析（PCA）**：对数据的协方差矩阵做特征分解，最大的特征值对应的方向就是数据变化最大的方向。Stable Diffusion 使用的"潜空间"就是通过类似方法找到的低维表示。

**Fokker-Planck 方程**：描述概率密度演化的方程中，扩散项涉及协方差矩阵，其特征值决定了不同方向上扩散的速度。

---

## 3.4 矩阵分解

### 3.4.1 SVD（奇异值分解）

任何矩阵 $A \in \mathbb{R}^{m \times n}$ 都可以分解为：

$$A = U \Sigma V^T$$

其中 $U \in \mathbb{R}^{m \times m}$ 和 $V \in \mathbb{R}^{n \times n}$ 是正交矩阵，$\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵（奇异值）。

**直觉**：SVD 告诉我们，任何线性变换都可以分解为"旋转→缩放→旋转"三步。

**在扩散模型中**：SVD 用于分析去噪网络在不同频率分量上的行为，理解为什么扩散模型能更好地处理低频结构。

### 3.4.2 Cholesky 分解

对称正定矩阵 $\Sigma$ 可以分解为：

$$\Sigma = LL^T$$

其中 $L$ 是下三角矩阵。

**在扩散模型中**：从多元高斯分布 $\mathcal{N}(\mu, \Sigma)$ 采样时，需要 Cholesky 分解：

1. 采样标准高斯向量 $\epsilon \sim \mathcal{N}(0, I)$
2. 计算 $x = \mu + L\epsilon$

这保证了 $\text{Cov}(x) = LL^T = \Sigma$。

在 DDPM 中，因为 $\Sigma = \sigma^2 I$ 是对角矩阵，所以 $L = \sigma I$，采样简化为 $x = \mu + \sigma\epsilon$。

---

## 3.5 线性变换与高斯分布

### 3.5.1 高斯分布的线性变换

如果 $x \sim \mathcal{N}(\mu, \Sigma)$，那么 $y = Ax + b$ 的分布是：

$$y \sim \mathcal{N}(A\mu + b, A\Sigma A^T)$$

**证明**：

均值：$\mathbb{E}[y] = A\mathbb{E}[x] + b = A\mu + b$

协方差：$\text{Cov}(y) = \mathbb{E}[(y - \bar{y})(y - \bar{y})^T] = A\mathbb{E}[(x-\mu)(x-\mu)^T]A^T = A\Sigma A^T$

**这是扩散模型中最常用的性质之一！**

### 3.5.2 DDPM 前向过程的矩阵视角

DDPM 的前向过程：

$$x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon$$

用矩阵表示（$\epsilon \sim \mathcal{N}(0, I)$）：

$$x_t = \sqrt{\alpha_t} \cdot I \cdot x_{t-1} + \sqrt{\beta_t} \cdot I \cdot \epsilon$$

利用高斯分布的线性变换性质：

$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$$

进一步，从 $x_0$ 直接到 $x_t$：

$$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

这里均值和方差的推导，本质上就是反复应用高斯分布的线性变换性质。

---

## 3.6 范数与距离

### 3.6.1 常用范数

| 范数 | 定义 | 用途 |
|------|------|------|
| $L^2$ 范数 | $\Vert\mathbf{x}\Vert_2 = \sqrt{\sum x_i^2}$ | 欧氏距离、MSE 损失 |
| $L^1$ 范数 | $\Vert\mathbf{x}\Vert_1 = \sum \vert x_i\vert$ | 稀疏性、鲁棒性 |
| $L^\infty$ 范数 | $\Vert\mathbf{x}\Vert_\infty = \max_i \vert x_i\vert$ | 最坏情况 |

### 3.6.2 Frobenius 范数

矩阵的 Frobenius 范数：

$$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

在计算两个高斯分布之间的 KL 散度时，均值之差的平方就用 Frobenius 范数（当均值是矩阵时）或 $L^2$ 范数（当均值是向量时）。

---

## 3.7 本章小结

| 线性代数工具 | 在扩散模型中的角色 |
|-------------|-------------------|
| 向量加法与数乘 | 前向过程：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| 内积与范数 | 损失函数：$\Vert\epsilon - \epsilon_\theta\Vert^2$ |
| 协方差矩阵 | 高斯分布的形状描述 |
| 矩阵的迹 | KL 散度的计算 |
| 行列式 | 概率密度的变量替换 |
| 特征分解 | PCA、协方差分析 |
| 高斯分布的线性变换 | 前向过程的推导 |
| Cholesky 分解 | 从高斯分布采样 |

---

## 练习与思考

1. **向量运算**：设 $x_t = \sqrt{0.9} \cdot x_0 + \sqrt{0.1} \cdot \epsilon$，其中 $x_0 = (1, 0)^T$，$\epsilon = (0, 1)^T$。计算 $x_t$ 和 $\|x_t\|^2$。

2. **协方差**：设 $x \sim \mathcal{N}(0, I)$，$y = Ax$，其中 $A = \text{diag}(2, 1)$。求 $y$ 的协方差矩阵。

3. **特征值**：求矩阵 $\begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ 的特征值和特征向量。（提示：特征值为 2 和 4）

4. **思考题**：为什么 DDPM 选择协方差矩阵为 $\beta_t I$（对角矩阵），而不是一般的协方差矩阵？这给模型带来了什么限制和便利？
