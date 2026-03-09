# MIT IAP 2026 · Lecture 5: Discrete Diffusion Models and Discrete Flow Matching

## 离散扩散模型与离散流匹配

---

## 一、本讲主线

前四讲都在连续空间里讨论生成模型：

- 数据是 $x \in \mathbb{R}^d$
- 动力学是 ODE 或 SDE
- 训练对象是向量场或 score

这一讲转向离散序列数据：

- 自然语言 token 序列
- 蛋白质序列
- 其他有限词表上的符号序列

核心问题是：

> 离散空间里没有通常意义下的 ODE，也没有通常意义下的 SDE。那 Flow Matching 和 Diffusion 的思想还能不能保留？

课件给出的答案是：

1. 把连续动力系统换成 **CTMC（Continuous-Time Markov Chain，连续时间马尔可夫链）**
2. 把连续空间里的向量场换成 **离散空间里的 rate matrix（速率矩阵）**
3. 把难以直接求的边际动力学，转化为一个每个 token 的 **posterior 分类问题**

所以，这一讲不是在讲“另一套完全不同的模型”，而是在讲：

> **如何把前几讲的生成建模思想推广到离散状态空间。**

---

## 二、课件开头的背景：为什么离散 diffusion 这两年又热起来

课件前几页先给了一个非常现实的背景：

- Google 的 `Gemini Diffusion`
- 一些离散 diffusion LLM 的新闻报道
- “可以达到很高 token/s”的速度卖点

这部分不是数学内容，但它说明了两个现实动机：

1. **文本本身就是离散对象**
   普通语言模型操作的是 token，不是连续像素。
2. **离散 diffusion 可能带来新的生成方式**
   不是固定左到右地吐 token，而是能并行、乱序地恢复序列。

课件还专门强调：

> Diffusion LLMs can generate text in arbitrary order.

这句话非常关键。它说明离散 diffusion 相比自回归语言模型，最大的潜在区别不是“网络结构变了”，而是：

> **生成顺序的约束变了。**

---

## 三、先把话说清楚：离散空间里没有“真正的 flow / diffusion”

课件第 5 页是整讲的纲领页，里面有一句很重要的话：

> There is no diffusion/SDE and also no flow/ODE in discrete space.

这句话的意思不是“离散扩散模型是错的”，而是：

- 在离散状态空间里，没法像连续空间那样定义速度向量 $u_t(x)$
- 也没法让状态沿一条几何曲线“平滑移动”
- 所以前几讲的 ODE/SDE 语言不能原样照搬

但课件紧接着说：

> Learning principles of flow matching and denoising can be generalized to discrete data.

也就是说，真正可迁移的是以下思想：

1. 设计一条从噪声到数据的概率路径
2. 构造匹配这条路径的动力学
3. 用可计算的监督目标训练神经网络

在离散情形里，对应的数学对象就是：

> **Continuous-time Markov chains (CTMCs).**

---

## 四、CTMC：离散空间里的连续时间动力系统

### 4.1 什么是 CTMC

CTMC 的状态空间是离散的，但时间是连续的。

一条 CTMC 轨迹长这样：

- 在某个状态上停留一段随机时长
- 然后在某个随机时刻跳到另一个状态
- 再停留
- 再跳

课件第 6 页的图就是这种“阶梯状”轨迹：

- 轨迹在时间轴上是分段常数
- 每次变化都对应一次瞬时跳跃

所以和 ODE 最大的区别是：

- ODE：状态连续移动
- CTMC：状态保持不变，偶尔突然跳变

### 4.2 两状态例子：$S = \\{a,b\\}$

课件第 7 页给了一个最简单的例子：

$$S = \{a,b\}$$

速率矩阵为

$$
Q =
\begin{pmatrix}
-\lambda & \lambda \\
\lambda & -\lambda
\end{pmatrix}
$$

这表示：

- 在状态 $a$ 时，以速率 $\lambda$ 跳到 $b$
- 在状态 $b$ 时，以速率 $\lambda$ 跳到 $a$

对这个系统，课件写出了转移概率矩阵：

$$
\begin{pmatrix}
p(X_{t+h}=a \mid X_t=a) & p(X_{t+h}=a \mid X_t=b) \\
p(X_{t+h}=b \mid X_t=a) & p(X_{t+h}=b \mid X_t=b)
\end{pmatrix}
=
\frac{1}{2}
\begin{pmatrix}
1 + e^{-2\lambda h} & 1 - e^{-2\lambda h} \\
1 - e^{-2\lambda h} & 1 + e^{-2\lambda h}
\end{pmatrix}
$$

当 $h \to \infty$ 时，上式收敛到

$$
\begin{pmatrix}
1/2 & 1/2 \\
1/2 & 1/2
\end{pmatrix}
$$

直觉上就是：

- 运行足够久以后，系统忘记起点
- 最后稳定在均匀分布上

这个例子的重要性在于它说明：

> **CTMC 像连续空间里的扩散一样，也可以把状态逐渐“混合”起来。**

### 4.3 邻接关系与“允许跳到哪里”

课件第 8 页用“neighbors”说明：

- 并不是所有状态之间都必须允许直接跳转
- 常常只允许从 $x$ 跳到某些邻居 $y$
- 对非邻居状态，速率可设为 0

也就是：

$$Q_t(y \mid x) = 0 \quad \text{if } x \text{ and } y \text{ are not neighbors}$$

这和图上的随机游走、图模型、离散状态搜索都很像。

### 4.4 一般 CTMC vs factorized CTMC

课件第 9 页给了非常重要的对比：

- **General CTMC**
  当前状态 $x$ 可以一下子跳到很多完全不同的状态 $y$
- **Factorized CTMC**
  每次只改动一个坐标或一个 token 位置

对于序列数据，factorized CTMC 更合理，因为样本可以写成

$$x = (x_1, \dots, x_d) \in \mathcal{V}^d$$

这里：

- $d$ 是序列长度
- $\mathcal{V}$ 是词表

factorized CTMC 的好处是：

1. 参数化简单
2. 每个位置可并行处理
3. 和语言模型“逐 token 输出 logits”的结构天然兼容

---

## 五、如何从 factorized CTMC 采样

课件第 10 页给出 `Algorithm 7`，标题是：

> Sampling from a Factorized CTMC Model (Euler / τ-leaping)

它是离散版的 Euler 采样。

### 5.1 输入和目标

算法输入是：

- factorized 的 rate network $Q_t^\theta$
- 初始分布 $p_{\text{init}}$
- 步数 $n$

目标是从 $X_0 \sim p_{\text{init}}$ 出发，逐步采样到 $X_1$。

### 5.2 每一步做什么

课件里的步骤可以整理成：

1. 令 $t \leftarrow 0$，步长 $h \leftarrow 1/n$
2. 采样 $X_0 \sim p_{\text{init}}$
3. 对每个时间步：
   对每个位置 $j$ 并行计算所有 token 的跳转率 $\{q_j(v)\}_{v \in \mathcal{V}}$
4. 对当前位置 $x = X_t^{(j)}$，构造一步 Euler 转移分布

$$
\tilde p_{j,t}(v \mid x) =
\begin{cases}
h\, q_j(v), & v \neq x \\
1 - h \sum_{v' \in \mathcal{V} \setminus \{x\}} q_j(v'), & v = x
\end{cases}
$$

5. 采样新的 token：

$$
X_{t+h}^{(j)} \sim \mathrm{Categorical}\big(\{\tilde p_{j,t}(v \mid x)\}_{v \in \mathcal{V}}\big)
$$

6. 更新时间 $t \leftarrow t+h$

### 5.3 这和连续 Euler 有什么对应关系

连续情形里：

$$x_{t+h} \approx x_t + h\,u_t(x_t)$$

离散情形里，没有“加一个向量”这回事，取而代之的是：

- 用速率决定短时间内“跳到哪儿”的概率
- 再从这个短时转移分布里采样

所以可以把它理解成：

> **连续 Euler 是沿速度方向前进一步；离散 Euler / τ-leaping 是按短时跳转概率随机更新一步。**

---

## 六、生成建模目标没有变：还是从噪声到数据

课件第 12 页重新画了一遍生成建模目标，只是把连续动力学换成了 CTMC：

- 数据分布：$p_{\text{data}}(z)$
- 初始分布：$p_{\text{init}}(z)$
- 目标：把噪声变成数据

也就是：

$$
X_0 \sim p_{\text{init}}
\quad \xrightarrow{\text{CTMC}} \quad
X_1 \sim p_{\text{data}}
$$

这里最重要的认知是：

> 生成模型的目标从来不是“连续”或“离散”本身，而是“把简单分布推送到复杂分布”。

只是：

- 连续空间用 ODE/SDE 完成推送
- 离散空间用 CTMC 完成推送

---

## 七、把 Lecture 2 的 Flow Matching 矩阵搬到离散空间

课件第 13 页先复习连续版 Flow Matching：

- Conditional probability path
- Marginal probability path
- Conditional vector field
- Marginal vector field
- Conditional FM loss
- Marginal FM loss

接着第 14 页给出离散版对应物：

| 连续版 | 离散版 |
|---|---|
| Conditional probability path | Conditional probability path |
| Marginal probability path | Marginal probability path |
| Conditional vector field | Conditional rate matrix |
| Marginal vector field | Marginal rate matrix |
| Conditional FM loss | Discrete FM loss |

所以这一讲真正做的事，可以概括成一句话：

> **把“向量场 + 连续性方程 + FM loss”整套 recipe，换成“速率矩阵 + Kolmogorov 方程 + DFM loss”。**

---

## 八、最核心的路径设计：Factorized Mixture Path

课件第 15 页开始进入本讲最重要的具体构造。

### 8.1 调度函数

取一个调度函数 $\kappa_t$，满足

$$
0 \le \kappa_t \le 1, \qquad \kappa_0 = 0, \qquad \kappa_1 = 1
$$

直觉上：

- $\kappa_t$ 越小，越接近噪声
- $\kappa_t$ 越大，越接近真实数据

### 8.2 条件概率路径

对每个位置独立地混合“噪声 token”和“真实 token”：

$$
p_t(x \mid z)
=
\prod_{j=1}^d
\Big[
(1-\kappa_t)p_{\text{init}}^{(j)}(x_j)

+ \kappa_t \delta_{z_j}(x_j)
\Big]
$$

这就是课件里写的 **factorized mixture path**。

解释如下：

- $p_{\text{init}}^{(j)}(x_j)$：第 $j$ 个位置的初始噪声边缘分布
- $\delta_{z_j}(x_j)$：把该位置钉死在真实 token $z_j$ 上
- $(1-\kappa_t)$：逐渐降低噪声权重
- $\kappa_t$：逐渐提高真实 token 权重

### 8.3 等价采样写法

课件还给出逐位置采样过程：

$$
m_j \sim \mathrm{Bernoulli}(\kappa_t), \qquad
\xi_j \sim p_{\text{init}}^{(j)}
$$

$$
x_j = m_j z_j + (1-m_j)\xi_j, \qquad j = 1,\dots,d
$$

虽然这里用了“加法”符号，但实际含义是二选一：

- $m_j = 1$ 时，直接取真实 token $z_j$
- $m_j = 0$ 时，取噪声 token $\xi_j$

### 8.4 为什么这条路径特别适合文本

因为它完全符合 token 的语义：

- 每个位置要么已经恢复正确
- 要么还保持噪声 / mask / 随机 token

对文本来说，这比连续高斯插值更自然。文本 token 之间没有“中间值”，所以最自然的路径不是“滑向正确值”，而是：

> **从不正确 token 直接跳成正确 token。**

---

## 九、课件中的 toy example：离散概率质量不是“流动”，而是“传送”

第 16 到 19 页都是示意图，用来强调一个和连续情形很不一样的点：

> Probability is teleported, not moved through space.

这句话非常值得展开。

连续空间里，如果你把一个高斯中心从左边移到右边，大家的直觉是：

- 概率质量沿空间平滑移动过去

但在离散空间里，状态之间并没有自然几何坐标。于是：

- 某个状态的概率减少
- 另一个状态的概率增加
- 中间并不存在“沿路径滑过去”的几何过程

更准确地说，离散路径的演化方式是：

> **概率质量在状态之间通过跳转重新分配。**

这也是为什么离散版不能再使用连续空间的散度、向量场和流映射语言，而要换成 CTMC 和 rate matrix。

---

## 十、离散版连续性方程：Kolmogorov Forward Equation

课件第 20 页给出离散 analogue of continuity equation：

若一个 CTMC 具有速率矩阵 $Q_t$，那么它沿着概率路径

$$
X_t \sim p_t \qquad (0 \le t \le 1)
$$

演化，当且仅当满足 **Kolmogorov Forward Equation (KFE)**：

$$
\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)
=
\sum_{y \in S} Q_t(x \mid y)\, p_t(y)
$$

这页课件把右边标成了：

- probability change
- net inflow

也就是：

> 某个状态 $x$ 的概率变化率，等于所有其他状态流入 $x$ 的净贡献。

如果把对角项单独拆出来，上式也可以写成更直观的“流入 - 流出”形式：

$$
\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)
=
\sum_{y \neq x} p_t(y)Q_t(x \mid y)
-
p_t(x)\sum_{y \neq x}Q_t(y \mid x)
$$

这和 Lecture 2 的连续性方程是完全平行的：

- 连续空间：$\partial_t p_t = -\mathrm{div}(p_t u_t)$
- 离散空间：$\partial_t p_t = \sum_y Q_t(x \mid y)p_t(y)$

---

## 十一、条件 rate matrix：如何让 CTMC 跟随 factorized mixture path

### 11.1 课件给出的精确公式

第 21 页给出条件速率矩阵：

$$
Q_t^z(y \mid x) = \big(Q_t^z(v_i, j \mid x_j)\big)_{v_i, j}
$$

并且

$$
Q_t^z(v_i, j \mid x_j)
=
\frac{\dot \kappa_t}{1-\kappa_t}
\big(\delta_{z_j}(v_i) - \delta_{x_j}(v_i)\big)
$$

这个式子其实已经把整件事说完了：

- $\delta_{z_j}(v_i)$：往正确 token 方向加质量
- $\delta_{x_j}(v_i)$：从当前 token 位置减去质量

### 11.2 按情况展开

课件把它展开成分段形式：

$$
Q_t^z(v_i, j \mid x_j)
=
\frac{\dot \kappa_t}{1-\kappa_t}
\begin{cases}
0, & x_j = z_j \\
1, & v_i = z_j,\; x_j \neq z_j \\
0, & v_i \neq z_j,\; x_j \neq z_j \\
-1, & v_i = x_j,\; x_j \neq z_j
\end{cases}
$$

更容易理解的说法是：

1. 如果当前位置已经正确，即 $x_j = z_j$
   速率为 0，不需要动。
2. 如果当前位置错误，即 $x_j \neq z_j$
   只允许往正确 token $z_j$ 跳。
3. 对当前错误 token 的对角项，会出现负的 outgoing rate。

课件右侧用红框总结得很直接：

- If current token correct, zero rate
- If incorrect, jump to correct token
- Outgoing rate from current token, if incorrect

### 11.3 为什么会有 $\frac{\dot\kappa_t}{1-\kappa_t}$

这个因子控制“恢复正确 token 的快慢”。

直觉上：

- $\kappa_t$ 增长越快，说明路径要求更快地靠近数据
- 那么 CTMC 就必须更快地把错误 token 改成正确 token

它本质上是：

> **路径速度** 在离散空间里的体现。

### 11.4 数值问题：$t \to 1$ 时速率会爆

第 22 到 23 页强调：

如果简单取

$$
\kappa_t = t
$$

那么

$$
\frac{\dot \kappa_t}{1-\kappa_t} = \frac{1}{1-t}
$$

在 $t \to 1$ 时会发散。

课件直接标出了：

> Rates explode at $t=1$

这意味着：

- 临近终点时，模型需要无限快地把所有错误 token 纠正掉
- 实现上常常要裁剪、截断或重新设计调度函数

这和连续 Flow Matching 里调度函数影响数值稳定性，是同一个问题。

---

## 十二、条件概率路径、条件 rate matrix 与“离散 score”

课件第 24 页把条件对象放到一张表里，总结为：

| 对象 | 记号 | 关键性质 | factorized mixture 公式 |
|---|---|---|---|
| Conditional Probability Path | $p_t(x \mid z)$ | 在 $p_{\text{init}}$ 和数据点 $z$ 之间插值 | factorized mixture |
| Conditional Rate Matrix | $Q_t^z(y \mid x)$ | CTMC follows conditional path | 上面的解析公式 |

这页的关键信息不是新增公式，而是强调：

> 在条件情形下，一切都是已知的、可写解析式的。

也就是说，给定：

- 当前噪声状态 $x$
- 目标终点 $z$

我们就知道：

1. 条件概率路径 $p_t(\cdot \mid z)$
2. 对应的条件 rate matrix $Q_t^z$

这和 Lecture 2 里“条件向量场可解析，边际向量场难计算”完全对应。

---

## 十三、边际 probability path 与边际 rate matrix

第 25 页先给出边际对象的总公式。

### 13.1 边际概率路径

$$
p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_{\text{data}}(z)
$$

这就是对条件路径按数据分布求平均。

### 13.2 边际 rate matrix

$$
Q_t(y \mid x)
=
\sum_{z \in S}
Q_t^z(y \mid x)\,
\frac{p_t(x \mid z)p_{\text{data}}(z)}{p_t(x)}
$$

这里的权重就是一个标准后验：

$$
p(z \mid x_t = x)
=
\frac{p_t(x \mid z)p_{\text{data}}(z)}{p_t(x)}
$$

这和连续 Flow Matching 中的边际向量场公式结构一模一样：

- 条件动力学知道终点
- 边际动力学不知道终点
- 所以要对所有可能终点取后验加权平均

---

## 十四、factorized mixture path 下的边际 rate matrix：未知量只剩 posterior

课件第 26 页是整讲最关键的桥。

它把条件 rate matrix

$$
Q_t^z(v_i, j \mid x_j)
=
\frac{\dot \kappa_t}{1-\kappa_t}
\big(\delta_{z_j}(v_i) - \delta_{x_j}(v_i)\big)
$$

转成边际 rate matrix：

$$
Q_t(y \mid x) = \big(Q_t(v_i, j \mid x)\big)_{v_i,j}
$$

$$
Q_t(v_i, j \mid x)
=
\frac{\dot \kappa_t}{1-\kappa_t}
\Big(
p_{1|t}(z_j = v_i \mid x) - \delta_{x_j}(v_i)
\Big)
$$

其中

$$
p_{1|t}(z_j = v_i \mid x)
$$

表示：

> 已知当前状态是 $x_t = x$，第 $j$ 个位置的终点 token 是 $v_i$ 的后验概率。

课件在这一页上明确标注：

- `Known terminal point!` 对应条件 rate matrix
- `Conditional probability! Only unknown!` 对应边际 rate matrix 中的 posterior

这页实际上在告诉你：

> 离散 Flow Matching 的难点，已经从“学整个 rate matrix”简化成了“学这个 posterior”。 

---

## 十五、Discrete Flow Matching loss：把生成建模问题变成分类问题

课件第 27 页标题就是 `Discrete Flow Matching loss`，并强调：

> Posterior probability network  
> Learn via classification.

这正是上一节的直接结果。

### 15.1 网络学什么

网络不直接输出完整边际 rate matrix，而是输出：

$$
p_{1|t}^\theta(z_j = v \mid x)
$$

也就是：

- 输入：当前噪声序列 $x$ 和时间 $t$
- 输出：每个位置上最终真实 token 的分类分布

### 15.2 损失函数

课件给出的 DFM 目标本质上是逐 token 的负对数似然：

$$
\mathcal{L}_{\mathrm{DFM}}(\theta)
=
\mathbb{E}_{z \sim p_{\text{data}},\; x \sim p_t(\cdot \mid z)}
\left[
\sum_{j=1}^d -\log p_{1|t}^\theta(z_j \mid x)_j
\right]
$$

更简单地说：

> 每个位置都在做一个分类任务，标签就是真实 token $z_j$。

这比连续 Flow Matching 中的均方误差回归更接近语言模型训练：

- 连续 FM：回归速度向量
- 离散 FM：分类终点 token

---

## 十六、训练算法（Algorithm 8）

课件第 28 页给出了完整训练算法 `Training factorized CTMC Model (Discrete Diffusion)`。

### 16.1 输入

算法需要：

- 数据集 $z \sim p_{\text{data}}$，其中 $z = (z_1,\dots,z_d) \in \mathcal{V}^d$
- 初始噪声边缘分布 $p_{\text{init}}^{(j)}$ on $\mathcal{V}$
- 调度函数 $\kappa_t \in [0,1]$
- posterior network $f_\theta$，输出每个位置关于词表的 logits
- 优化器 `OPT`

### 16.2 训练步骤

算法流程是：

1. 采样一个真实序列 $z \sim p_{\text{data}}$
2. 采样时间 $t \sim \mathrm{Unif}[0,1]$，并计算 $\kappa \leftarrow \kappa_t$
3. 采样 noisy state $x \sim p_t(\cdot \mid z)$：
   对每个位置 $j$ 并行地做

$$
m_j \sim \mathrm{Bernoulli}(\kappa), \qquad \xi_j \sim p_{\text{init}}^{(j)}
$$

$$
x_j \leftarrow m_j z_j + (1-m_j)\xi_j
$$

4. 网络输出 logits：

$$
\ell_j(\cdot) \leftarrow f_\theta(x,t)_j
\qquad \Longrightarrow \qquad
p_{1|t}^\theta(v \mid x)_j = \mathrm{Softmax}(\ell_j)(v)
$$

5. 计算 DFM 损失：

$$
\mathcal{L}_{\mathrm{DFM}}(\theta)
\leftarrow
\sum_{j=1}^d \Big[-\log p_{1|t}^\theta(z_j \mid x)_j \Big]
$$

6. 用优化器更新参数：

$$
\theta \leftarrow \mathrm{OPT.STEP}\big(\nabla_\theta \mathcal{L}_{\mathrm{DFM}}(\theta)\big)
$$

### 16.3 这和普通语言模型训练有多像

非常像。

区别只在于：

- 普通自回归 LM 的输入是前缀上下文
- 这里的输入是一个“被随机破坏的整句”

但输出头几乎一样：

- 对每个位置输出词表 logits
- 对真实 token 做交叉熵

所以课件这一部分的隐含信息是：

> 离散 diffusion / discrete FM 在工程上可以自然复用语言模型的 token 分类头。

---

## 十七、Mask Diffusion Language Models

第 29 页开始，课件把上述框架应用到最自然的文本场景。

### 17.1 为什么引入 `[MASK]`

课件写得很直接：

- 在词表中引入一个新 token：`[MASK]`
- 它表示“真实 token 被遮住了”

然后把初始分布设为：

$$
\delta_{[\mathrm{MASK}]}
$$

也就是说，开始时所有位置都是 `[MASK]`。

### 17.2 这有什么好处

这样做以后，factorized mixture path 的语义变得特别清晰：

- 早期：句子里大部分位置还是 `[MASK]`
- 中期：一部分位置已经恢复成真实词
- 后期：几乎整句都恢复出来

比起“从均匀随机 token 分布出发”，全 mask 初始分布更符合人类对“逐步填空”的直觉。

因此，masked diffusion language model 可以理解成：

> **把 BERT 风格的 mask 恢复，变成一个连续时间、逐步去噪的生成过程。**

---

## 十八、采样演示：从 fully masked 到完整文本

第 31 到 35 页是课件里最直观的 demo。

### 18.1 起点

第 31 页写的是：

> Start with fully masked (initial distribution)

也就是整个句子一开始都是 `[MASK]`。

### 18.2 中间阶段

后面几页展示随着时间推进：

- $t=0.3$ 时，只恢复出少量词，句子还很破碎
- $t=0.6$ 时，更多高概率词被填回来，语义轮廓开始出现
- $t=0.8$ 时，大多数句子结构已经可读
- $t=1.0$ 时，整段话完整恢复

课件展示的例子是《百年孤独》开头那段著名英文句子：

> Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember ...

### 18.3 这个 demo 想说明什么

它想说明离散 diffusion 的生成过程不是：

- 固定从左到右写句子

而是：

- 整个句子同时存在
- 只是很多位置一开始是未知 / 被遮住的
- 模型逐步在全局上下文下把这些位置恢复出来

这也是为什么课件在前面一再强调：

> text in arbitrary order

从用户体验上看，这意味着它天然适合：

- 局部编辑
- 插空补全
- 改写某一小段而保持其他位置不变

---

## 十九、离散 diffusion vs 自回归语言模型

课件第 37 页给出了一张非常平衡的对比表。

### 19.1 优点

1. **Generate Multiple Tokens in Parallel**
   可以同时更新多个 token，潜在上更快。
2. **Generate Tokens in any order**
   适合文本编辑，不受左到右顺序约束。
3. **New probability paths**
   可以重新设计路径，而不必局限于固定生成顺序。

第三点特别值得注意。

自回归语言模型的“概率路径”几乎是被生成顺序固定死的：

- 第 1 个 token
- 第 2 个 token
- 第 3 个 token
- …

而离散 diffusion 允许你设计别的路径，比如：

- mask path
- mixture path
- 更偏向语义结构的 path

这为未来模型设计留下了空间。

### 19.2 缺点

1. **No KV caching**
   推理时不像自回归 Transformer 那样直接受益于 KV cache。
2. **Need to learn how to generate tokens in any order**
   任务本身更难，因为模型必须学会无序恢复。
3. **Autoregressive order makes semantic sense**
   左到右顺序本身就是一种很强的自然语言归纳偏置。

课件最后一句问得很直接：

> Is it worth it?

这是个非常诚实的问题。也就是说，离散 diffusion 不只是“更酷的新方法”，它是否真比自回归更划算，要看：

- 推理效率
- 训练难度
- 编辑能力
- 最终效果

---

## 二十、和连续 Flow Matching 的统一理解

课件第 38 到 39 页又把连续版和离散版矩阵并排放了一次，目的是强调：

- 虽然对象变了
- 但 recipe 基本没变

把两讲对照起来看：

| 连续空间 | 离散空间 |
|---|---|
| Probability path $p_t$ | Probability path $p_t$ |
| Vector field $u_t$ | Rate matrix $Q_t$ |
| Continuity equation | Kolmogorov Forward Equation |
| Conditional FM loss | Discrete FM / posterior classification loss |

所以第五讲真正新增的不是“新的哲学”，而是“新的动力学实现”。

更抽象地说：

> Flow Matching 本质上是一种“给定概率路径，学习与之匹配的 Markov 动力学”的方法。

---

## 二十一、最后一页的高层总结：Generator Matching

课件第 40 页抛出一个更一般的框架：

> Generator Matching: Generative Modeling with Arbitrary Markov Processes

这句话的意思是：

- ODE 是一种 Markov process
- SDE 是一种 Markov process
- CTMC 也是一种 Markov process

所以从更高层看，Flow Matching 的核心并不依赖某种具体动力学，而依赖：

1. 你先指定一个目标概率路径
2. 再指定一个可学习的 Markov generator
3. 用匹配原则把 generator 训练到能实现这条路径

这就是为什么 Lecture 5 在课程结构里很重要。

它说明：

> **Flow Matching 不是“图像扩散里的一个技巧”，而是一个可以跨连续 / 离散空间复用的生成学习原则。**

---

## 二十二、整节课按页序压缩回顾

如果按照课件页序，只保留主干逻辑，这一讲可以压缩成：

1. 离散 diffusion LLM 变热，因为文本是天然离散的，而且它支持任意顺序生成。
2. 离散空间里没有 ODE/SDE，所以要改用 CTMC。
3. CTMC 的动力学对象是 rate matrix，采样可用 Euler / τ-leaping。
4. 为离散 Flow Matching 设计一条 factorized mixture path：

$$
p_t(x \mid z)=\prod_j\big[(1-\kappa_t)p_{\text{init}}^{(j)}(x_j)+\kappa_t\delta_{z_j}(x_j)\big]
$$

5. 用 KFE 取代连续性方程。
6. 条件 rate matrix 可解析写出：

$$
Q_t^z(v_i,j \mid x_j)=\frac{\dot\kappa_t}{1-\kappa_t}\big(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\big)
$$

7. 边际 rate matrix 中唯一未知量是 posterior：

$$
Q_t(v_i,j \mid x)
=
\frac{\dot\kappa_t}{1-\kappa_t}
\Big(p_{1|t}(z_j=v_i \mid x)-\delta_{x_j}(v_i)\Big)
$$

8. 因而训练问题变成 posterior 分类问题：

$$
\mathcal{L}_{\mathrm{DFM}}(\theta)
=
\sum_{j=1}^d -\log p_{1|t}^\theta(z_j \mid x)_j
$$

9. 在文本里，最自然的初始分布就是全 `[MASK]`。
10. 这给出了一种不同于自回归 LM 的生成范式：全局、并行、可编辑，但也更难训练和优化。

---

## 二十三、和前四讲的连接

把五讲连起来看，逻辑非常完整：

1. Lecture 1：ODE / SDE 生成模型的基本语言
2. Lecture 2：Flow Matching 的训练原理
3. Lecture 3：Score Matching、随机采样和 CFG
4. Lecture 4：潜空间和 DiT 架构
5. Lecture 5：把整套思想扩展到离散状态空间

因此第五讲的真正结论不是“文本也能做 diffusion”，而是：

> **只要你能写出合适的概率路径和 Markov generator，Flow Matching 的思路就可以超越连续图像空间。**

---

## 二十四、复习时最该记住的六句话

1. 离散空间里没有传统 ODE/SDE，所以要用 CTMC。
2. CTMC 的核心对象是速率矩阵 $Q_t(y \mid x)$。
3. factorized mixture path 让“噪声 token 和真实 token 的混合”变得可计算。
4. KFE 是离散空间里对应连续性方程的守恒定律。
5. 边际 rate matrix 的难点被压缩成 posterior $p_{1|t}(z_j=v \mid x)$。
6. 因而 discrete flow matching 的训练最终就是一个逐 token 分类问题。
