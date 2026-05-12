# 第4章 概率论与数理统计：随机世界的语言

> 扩散模型本质上是一个概率模型——它用概率分布来描述数据如何变成噪声，又如何从噪声恢复为数据。高斯分布、条件概率、贝叶斯定理，这些概念构成了扩散模型的数学骨架。

---

## 4.1 随机变量与概率分布

### 4.1.1 随机变量

随机变量是一个"取值不确定"的量。我们用大写字母 $X$ 表示随机变量，小写字母 $x$ 表示它的具体取值。

- **离散随机变量**：取值是有限个或可数个（如掷骰子的结果）
- **连续随机变量**：取值可以充满某个区间（如人的身高）

在扩散模型中，图像 $x_0$ 和噪声 $\epsilon$ 都是连续随机变量。

### 4.1.2 概率密度函数（PDF）

对于连续随机变量 $X$，概率密度函数 $p(x)$ 满足：

$$P(a \leq X \leq b) = \int_a^b p(x) dx$$

**关键性质**：

1. $p(x) \geq 0$（概率不能为负）
2. $\int_{-\infty}^{+\infty} p(x) dx = 1$（总概率为 1）

**注意**：$p(x)$ 本身不是概率！$p(x)$ 可以大于 1。只有积分才是概率。

### 4.1.3 累积分布函数（CDF）

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} p(t) dt$$

CDF 和 PDF 的关系：$p(x) = F'(x)$

---

## 4.2 期望、方差与协方差

### 4.2.1 期望（均值）

期望是随机变量的"平均值"：

$$\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot p(x) dx$$

**期望的线性性质**（极其重要）：

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

**注意**：这个性质不需要 $X$ 和 $Y$ 独立！

### 4.2.2 方差

方差衡量随机变量的"分散程度"：

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**计算技巧**：$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ 通常比定义式更方便计算。

**方差的性质**：

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

当 $X, Y$ 独立时：$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 4.2.3 协方差

协方差衡量两个随机变量的"线性关联程度"：

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

- $\text{Cov}(X, Y) > 0$：$X$ 和 $Y$ 倾向于同向变化
- $\text{Cov}(X, Y) < 0$：$X$ 和 $Y$ 倾向于反向变化
- $\text{Cov}(X, Y) = 0$：$X$ 和 $Y$ 不线性相关（但不一定独立！）

### 4.2.4 扩散模型中的应用

在 DDPM 的前向过程中：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

计算 $x_t$ 的均值和方差：

$$\mathbb{E}[x_t] = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \mathbb{E}[\epsilon] = \sqrt{\bar{\alpha}_t} x_0$$

$$\text{Var}(x_t) = (1-\bar{\alpha}_t) \text{Var}(\epsilon) = (1-\bar{\alpha}_t) I$$

这里用到了：$\mathbb{E}[\epsilon] = 0$，$\text{Var}(\epsilon) = I$，以及 $x_0$ 给定时是常数。

---

## 4.3 高斯分布——扩散模型的基石

### 4.3.1 一维高斯分布

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| 参数 | 含义 |
|------|------|
| $\mu$ | 均值——分布的中心位置 |
| $\sigma^2$ | 方差——分布的分散程度 |
| $\sigma$ | 标准差 |

**高斯分布的形状**：钟形曲线，关于 $\mu$ 对称，$\sigma$ 越大越"胖"，$\sigma$ 越小越"瘦"。

**68-95-99.7 法则**：
- 约 68% 的数据落在 $[\mu - \sigma, \mu + \sigma]$
- 约 95% 的数据落在 $[\mu - 2\sigma, \mu + 2\sigma]$
- 约 99.7% 的数据落在 $[\mu - 3\sigma, \mu + 3\sigma]$

### 4.3.2 多维高斯分布

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu} \in \mathbb{R}^d$ 是均值向量，$\Sigma \in \mathbb{R}^{d \times d}$ 是协方差矩阵。

**特殊情况：各维度独立**（$\Sigma = \sigma^2 I$）：

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 I) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma^2}\right)$$

当协方差矩阵是对角矩阵时，各维度独立，联合分布等于各维度边缘分布的乘积。**DDPM 的每一步都假设噪声是各维度独立的高斯分布**，即 $\Sigma = \beta_t I$。

### 4.3.3 高斯分布的四个关键性质

**性质 1：高斯的线性变换仍是高斯**

如果 $x \sim \mathcal{N}(\mu, \Sigma)$，则 $y = Ax + b \sim \mathcal{N}(A\mu + b, A\Sigma A^T)$。

**性质 2：独立高斯的和仍是高斯**

如果 $x \sim \mathcal{N}(\mu_1, \Sigma_1)$ 与 $y \sim \mathcal{N}(\mu_2, \Sigma_2)$ 独立，则 $x + y \sim \mathcal{N}(\mu_1 + \mu_2, \Sigma_1 + \Sigma_2)$。

**这是 DDPM 前向过程推导的核心！** 递推展开时，每一步加入的独立高斯噪声可以合并为一个高斯噪声：

$$\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1} \sim \mathcal{N}(0, (\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t))I)$$

**性质 3：高斯的条件分布仍是高斯**

如果 $(x, y)$ 服从联合高斯分布，那么条件分布 $p(x|y)$ 也是高斯分布。这是推导 DDPM 反向过程后验分布的基础。

**性质 4：高斯的边缘分布仍是高斯**

如果 $(x, y)$ 服从联合高斯分布，那么边缘分布 $p(x) = \int p(x, y) dy$ 也是高斯分布。

### 4.3.4 为什么高斯分布如此重要？

1. **中心极限定理**：大量独立随机变量之和趋向于高斯分布。这意味着"累积噪声"自然趋向高斯分布
2. **数学上的方便性**：高斯分布的线性变换、条件分布、边缘分布都是高斯——这让推导变得可行
3. **最大熵原理**：在只知道均值和方差的条件下，高斯分布是信息熵最大的分布——它对未知信息做了最少的假设
4. **物理上的合理性**：布朗运动（物理中的扩散现象）的分布就是高斯分布

---

## 4.4 条件概率与贝叶斯定理

### 4.4.1 条件概率

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

读作"在 $B$ 发生的条件下，$A$ 发生的概率"。

**连续版本**：

$$p(x|y) = \frac{p(x, y)}{p(y)}$$

### 4.4.2 贝叶斯定理

$$p(x|y) = \frac{p(y|x) \cdot p(x)}{p(y)}$$

或等价地：

$$p(x|y) = \frac{p(y|x) \cdot p(x)}{\int p(y|x) \cdot p(x) dx}$$

**贝叶斯定理的含义**：

| 术语 | 公式 | 含义 |
|------|------|------|
| 后验概率 | $p(x \mid y)$ | 观测到 $y$ 后，对 $x$ 的信念 |
| 似然 | $p(y \mid x)$ | $x$ 为真时，观测到 $y$ 的概率 |
| 先验概率 | $p(x)$ | 观测前对 $x$ 的信念 |
| 证据 | $p(y)$ | 观测到 $y$ 的总概率（归一化常数） |

**贝叶斯定理的直觉**：后验 = 似然 × 先验 / 证据。观测到新数据后，我们用"数据与模型的吻合程度"（似然）来更新"原来的信念"（先验），得到"更新后的信念"（后验）。

### 4.4.3 贝叶斯定理在 DDPM 中的核心应用

**推导反向过程的后验分布** $q(x_{t-1} | x_t, x_0)$：

已知：
- $q(x_t | x_{t-1})$：前向转移分布（已知）
- $q(x_{t-1} | x_0)$：前向边缘分布（已知）
- $q(x_t | x_0)$：前向边缘分布（已知）

利用贝叶斯定理：

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由于马尔可夫性：$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

**这个公式是 DDPM 最重要的推导起点！** 它告诉我们：给定当前噪声状态 $x_t$ 和原始数据 $x_0$，前一步 $x_{t-1}$ 的分布是什么。

由于等式右边的三个分布都是高斯分布，而高斯的贝叶斯后验仍是高斯，所以 $q(x_{t-1}|x_t, x_0)$ 也是高斯分布——我们可以精确计算它的均值和方差。

### 4.4.4 高斯贝叶斯的计算

设三个高斯分布为：

$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$$
$$q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1}) I)$$
$$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

对贝叶斯公式取对数：

$$\log q(x_{t-1}|x_t, x_0) = \log q(x_t|x_{t-1}) + \log q(x_{t-1}|x_0) - \log q(x_t|x_0) + C$$

展开每个高斯分布的对数（只保留与 $x_{t-1}$ 有关的项）：

$$\log q(x_t|x_{t-1}) \propto -\frac{1}{2\beta_t}\|x_t - \sqrt{\alpha_t}x_{t-1}\|^2$$

$$\log q(x_{t-1}|x_0) \propto -\frac{1}{2(1-\bar{\alpha}_{t-1})}\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\|^2$$

$$\log q(x_t|x_0) \text{ 与 } x_{t-1} \text{ 无关}$$

合并后配方，得到：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

其中：

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0$$

**这个推导过程是整个 DDPM 的数学核心。** 它综合运用了：
- 条件概率与贝叶斯定理
- 高斯分布的性质
- 对数运算
- 配方（完成平方）

---

## 4.5 独立性与条件独立性

### 4.5.1 独立

$X$ 和 $Y$ 独立，当且仅当：

$$p(x, y) = p(x) \cdot p(y)$$

等价地：$p(x|y) = p(x)$，即知道 $Y$ 不改变对 $X$ 的信念。

### 4.5.2 条件独立

$X$ 和 $Y$ 在给定 $Z$ 的条件下独立，记作 $X \perp\!\!\!\perp Y | Z$：

$$p(x, y | z) = p(x | z) \cdot p(y | z)$$

### 4.5.3 马尔可夫性

马尔可夫性是条件独立性的一个特例。对于序列 $x_0, x_1, \ldots, x_T$，如果：

$$p(x_t | x_{t-1}, x_{t-2}, \ldots, x_0) = p(x_t | x_{t-1})$$

则称这个序列具有马尔可夫性——**未来只依赖现在，不依赖过去**。

**DDPM 的前向过程就是马尔可夫链**：每一步 $x_t$ 只依赖前一步 $x_{t-1}$，不依赖更早的历史。这使得前向过程的联合分布可以分解为：

$$q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

---

## 4.6 大数定律与中心极限定理

### 4.6.1 大数定律

大数定律说：当样本量 $n$ 趋向无穷时，样本均值趋向于总体均值。

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{P} \mathbb{E}[X]$$

**在扩散模型中**：损失函数的期望 $\mathbb{E}_{x_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ 无法精确计算，但可以通过小批量样本的平均值来近似——这就是大数定律的保证。

### 4.6.2 中心极限定理

中心极限定理说：大量独立同分布的随机变量之和（标准化后）趋向于标准高斯分布。

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**在扩散模型中的意义**：前向过程中，每一步加入的噪声是独立的高斯噪声。经过多步累积后，总噪声自然趋向于高斯分布——这正是中心极限定理的体现。即使单步噪声不是高斯的，多步累积后也会趋向高斯。

---

## 4.7 蒙特卡洛方法

### 4.7.1 用采样近似期望

很多时候，我们需要计算期望 $\mathbb{E}_{x \sim p}[f(x)]$，但无法解析计算。蒙特卡洛方法的核心思想：**用样本平均近似期望**。

$$\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x^{(i)}), \quad x^{(i)} \sim p$$

### 4.7.2 在扩散模型中的应用

**训练时的损失计算**：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] \approx \frac{1}{B}\sum_{i=1}^{B}\|\epsilon^{(i)} - \epsilon_\theta(x_t^{(i)}, t^{(i)})\|^2$$

其中 $B$ 是批量大小。这就是蒙特卡洛近似。

**采样时的去噪过程**：从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪得到 $x_0$。每次采样都是一次蒙特卡洛试验。

---

## 4.8 本章小结

| 概率工具 | 在扩散模型中的角色 |
|---------|-------------------|
| 随机变量与概率分布 | 描述数据和噪声的随机性 |
| 期望与方差 | 计算 $x_t$ 的均值和方差 |
| 高斯分布 | 每一步噪声的分布 |
| 高斯的线性变换 | 推导前向过程的分布 |
| 独立高斯的和 | 合并多步噪声为一步 |
| 条件概率与贝叶斯定理 | 推导反向过程的后验分布 |
| 马尔可夫性 | 前向过程的链式分解 |
| 中心极限定理 | 解释为什么高斯分布如此自然 |
| 蒙特卡洛方法 | 近似计算期望和损失 |

---

## 练习与思考

1. **高斯分布**：设 $X \sim \mathcal{N}(0, 1)$，$Y = 2X + 3$。求 $Y$ 的分布。

2. **独立高斯的和**：设 $\epsilon_1 \sim \mathcal{N}(0, \sigma_1^2)$ 和 $\epsilon_2 \sim \mathcal{N}(0, \sigma_2^2)$ 独立。证明 $\sigma_1 \epsilon_1 + \sigma_2 \epsilon_2 \sim \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$。

3. **贝叶斯定理**：设 $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{0.9} x_{t-1}, 0.1)$，$q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{0.8} x_0, 0.2)$，$q(x_t | x_0) = \mathcal{N}(\sqrt{0.72} x_0, 0.28)$。用贝叶斯定理求 $q(x_{t-1} | x_t, x_0)$ 的均值和方差。

4. **思考题**：为什么扩散模型选择高斯噪声而不是其他分布？如果用均匀噪声会怎样？（提示：考虑中心极限定理和高斯分布的数学性质）
