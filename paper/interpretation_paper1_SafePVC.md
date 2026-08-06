# Paper 1 详细解读：SafePVC — 基于随机障碍证书的视觉神经网络控制系统可证明概率安全控制器合成

> **论文标题**: Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems
>
> **作者**: Anonymous (DAC'26 投稿)
>
> **发表**: DAC 2026
>
> **核心贡献**: 首次联合使用强化学习 (PPO) 和随机障碍证书 (SBC)，为基于视觉的神经网络控制系统合成高性能控制器，并提供**无限时间范围**内的形式化概率安全下界 $P(\text{safe}) \geq p$。通过反例引导的交替优化迭代精炼控制器和证书网络。

---

## 一、问题形式化

### 1.1 系统模型

考虑**离散时间动力系统**：

$$
\boxed{s_{t+1} = f(s_t, u_t), \quad o_t = g(s_t, z_t), \quad u_t = \pi(o_t)}
$$

初始状态 $s_0 \in \mathcal{S}_0$。

### 1.2 全部符号定义

| 符号 | 含义 | 维度/空间 | 已知? |
|------|------|----------|-------|
| $s_t \in \mathcal{S}$ | 系统状态 | $\mathcal{S} \subseteq \mathbb{R}^m$，紧凑 | 执行时不可观测 |
| $u_t \in \mathcal{U}$ | 控制动作 | $\mathcal{U} \subseteq \mathbb{R}^n$ | — |
| $o_t \in \mathcal{O}$ | 视觉观测（灰度图像） | $\mathcal{O} \subseteq \mathbb{R}^{H \times W}$ | 执行时唯一可观测 |
| $z_t \in \mathcal{Z}$ | 未观测的环境扰动 | $\mathcal{Z} \subseteq \mathbb{R}^p$ | **不可观测** |
| $f: \mathcal{S} \times \mathcal{U} \to \mathcal{S}$ | 系统动力学 | $\mathbb{R}^m \times \mathbb{R}^n \to \mathbb{R}^m$ | **已知** |
| $g: \mathcal{S} \times \mathcal{Z} \to \mathcal{O}$ | 观测映射 | — | **未知** |
| $\pi: \mathcal{O} \to \mathcal{U}$ | 视觉控制策略 | — | **未知（待合成）** |
| $m, n, p \in \mathbb{N}$ | 状态/控制/扰动维度 | — | — |
| $H, W \in \mathbb{N}$ | 图像高度和宽度 | — | — |

**关键设定**：
1. 动力学 $f$ **已知**（如飞机滑行模型、车辆制动模型）
2. 观测函数 $g$ **未知**（真实物理映射复杂且不可靠）
3. 执行时仅观测 $o_t$ 可知，状态 $s_t$ 和环境 $z_t$ 不可观测

### 1.3 Assumption 1 — 环境扰动的状态偏差模型

设 $z_0$ 为**参考环境条件**。存在映射 $F: \mathcal{S} \times \mathcal{Z} \to \mathcal{S}$，由 $f, g, \pi$ 决定，使得给定 $s \in \mathcal{S}$ 和 $z \in \mathcal{Z}$，下一状态为：

$$
\boxed{s' = F(s, z)}
$$

定义**状态偏差**（环境扰动导致的状态偏移）：

$$
\boxed{\Delta s = F(s, z) - F(s, z_0) \in \Omega_\Delta \subset \mathbb{R}^m}
$$

其中 $\Delta s$ 是随机变量，具有 Borel 概率测度：

$$
\boxed{\Delta s \sim \mu, \quad \mu \in \mathcal{P}(\Omega_\Delta)}
$$

**关键假设**：$\Delta s$ 与当前状态 $s$ **独立**。

闭环动力学重写为：

$$
\boxed{s' = \tilde{F}(s, z_0, \Delta s) = F(s, z_0) + \Delta s}
$$

这是**离散时间 Markov 过程**，具有概率测度 $\mathbb{P}_{s_0}$ 和期望算子 $\mathbb{E}_{s_0}$。

### 1.4 概率安全定义 (Definition 2.1)

$$
\boxed{\mathbb{P}_{s_0}\big[\text{Safe}(X_u)\big] \geq p}
$$

| 符号 | 定义 | 含义 |
|------|------|------|
| $X_u \subseteq \mathcal{S}$ | Borel 可测不安全集 | 状态空间中"出事"的区域 |
| $p \in [0, 1)$ | 概率阈值 | 安全概率的下界 |
| $\text{Safe}(X_u)$ | $\{(s_t, \Delta_s)_{t \in \mathbb{N}_0} \mid \forall t \in \mathbb{N}_0, s_t \notin X_u\}$ | **永远不进入** $X_u$ 的轨迹集合 |

**重要**：这是**无限时间范围**的安全定义（$\forall t \in \mathbb{N}_0$），比 SPVT 的 K 步安全更强。

---

## 二、核心理论：随机障碍证书 (SBC)

### 2.1 Theorem 2.2 — SBC 四个条件

连续函数 $B: \mathcal{S} \to \mathbb{R}$ 是关于 $\mathcal{S}_0, X_u, p$ 的 SBC，需满足：

$$
\boxed{\begin{aligned}
\text{(i)} &\quad B(s) \geq 0, && \forall s \in \mathcal{S} && \text{—— 全局非负性} \\
\text{(ii)} &\quad B(s) \leq 1, && \forall s \in \mathcal{S}_0 && \text{—— 初始有界性} \\
\text{(iii)} &\quad B(s) \geq \frac{1}{1-p}, && \forall s \in X_u && \text{—— 不安全惩罚} \\
\text{(iv)} &\quad B(s) \geq \mathbb{E}_{\Delta s \sim \mu}\big[B(\tilde{F}(s, z_0, \Delta s))\big] + \epsilon, && \forall s \in \mathcal{S} \setminus X_u \text{ 且 } B(s) \leq \frac{1}{1-p} && \text{—— 期望递减（鞅条件）}
\end{aligned}}
$$

其中 $\epsilon > 0$ 是严格递减边际。

### 2.2 每个条件的直观解释

| 条件 | 直观 | 为什么需要 |
|------|------|-----------|
| (i) | B 值始终 $\geq 0$ | 障碍函数不能取负——负值没有概率解释 |
| (ii) | 从 $\mathcal{S}_0$ 出发时 $B \leq 1$ | 设置"起点"——初始障碍值有上界 |
| (iii) | 不安全集 $B \geq 1/(1-p)$ | 门槛——进入不安全集需要巨大的 B 值 |
| (iv) | 安全区域内 $B(s_t)$ 是**超鞅** | 核心——B 的期望递减，从 1 开始达不到 $1/(1-p)$ |

### 2.3 安全概率下界的推导（鞅理论 + 可选停止定理）

**推导步骤**：

**Step 1**：定义停止时间 $\tau = \min\{t \in \mathbb{N}_0 : s_t \in X_u\}$（首次进入不安全集的时间）。

**Step 2**：由条件 (iv)，在安全区域 $\mathcal{S} \setminus X_u$ 内 $B(s_t)$ 是**超鞅 (supermartingale)**：
$$
\mathbb{E}[B(s_{t+1}) \mid s_t] \leq B(s_t) - \epsilon < B(s_t)
$$

**Step 3**：由**可选停止定理 (Optional Stopping Theorem)**，对于超鞅 $B(s_t)$ 和有界停止时间 $\tau \wedge T$：
$$
\mathbb{E}[B(s_{\tau \wedge T})] \leq B(s_0)
$$

**Step 4**：令 $T \to \infty$，考虑事件 $\{\tau < \infty\}$（最终进入不安全集）：
$$
\begin{aligned}
B(s_0) &\geq \mathbb{E}[B(s_\tau) \mid \tau < \infty] \cdot \mathbb{P}(\tau < \infty) \\
&\geq \frac{1}{1-p} \cdot \mathbb{P}(\tau < \infty) \quad \text{（由条件 iii）}
\end{aligned}
$$

**Step 5**：由条件 (ii)，$B(s_0) \leq 1$：
$$
1 \geq B(s_0) \geq \frac{1}{1-p} \cdot \mathbb{P}(\tau < \infty)
$$

因此：
$$
\mathbb{P}(\tau < \infty) \leq 1 - p \quad \Longrightarrow \quad \boxed{\mathbb{P}(\text{永远安全}) = \mathbb{P}(\tau = \infty) \geq p}
$$

**这就是 SafePVC 的核心数学保证**：如果 SBC 存在并被验证，则从 $\mathcal{S}_0$ 中任意初始状态出发，系统以至少概率 $p$ **永远不进入**不安全集。

---

## 三、框架总览：SafePVC 双组件架构

SafePVC 由两大组件构成：**左侧**（观测近似器）和**右侧**（控制器生成器），中间以 VCLS（可验证闭环系统）桥接。

### 3.1 架构图

```
┌──────────────────────────────┐     ┌─────────────────────────────────────┐
│   组件一：观测近似器           │     │   组件二：控制器生成器                │
│   (Observation Approximator)  │     │   (Controller Generator)             │
│                              │     │                                      │
│  ① cGAN 感知模型             │     │  ⑥ 神经 SBC 合成                     │
│     G(s,z) → 伪图像          │     │     B(s) 网络 + 四条件损失            │
│       ↓                      │     │       ↓                              │
│  ② MLP 蒸馏 + Lipschitz      │     │  ⑦ 形式化验证                        │
│     g(s,z): 紧凑可验证模型     │     │     α,β-CROWN (条件ii,iii)           │
│       ↓                      │     │     Theorem 3.1 (条件iv)             │
│  ③ PPO 预训练控制器 π₀       │     │       ↓                              │
│     RL 在蒸馏模型上训练        │     │  ⑧ 反例收集 + 交替优化               │
│       ↓                      │     │     SBC 每轮更新                     │
│  ④ 数据驱动扰动估计           │     │     π 每 10 轮更新                   │
│     Δs ∼ μ 统计分布           │     │       ↓                              │
│       ↓                      │     │  ⑨ 更新 Δs 分布（闭环变化后）         │
│  ⑤ VCLS = g ∘ π₀            │     │       ↓                              │
│     可验证闭环系统             │     │  ⑩ 输出：安全 π + 有效 SBC           │
└──────────────────────────────┘     └─────────────────────────────────────┘
```

---

## 四、组件一：可验证闭环系统 (VCLS) 构建

### 4.1 Step 1: cGAN 感知模型

**目标**：学习从 $(s, z)$ 到观测 $o$ 的映射（近似未知的 $g$）。

**生成器** $G(s, z)$：输入状态 $s$ 和环境扰动 $z$，输出伪图像。
**判别器** $D(o, s)$：判断图像 $o$ 是否与状态 $s$ 匹配（条件 GAN）。

**cGAN 训练目标**：

$$
\boxed{\min_G \max_D \; \mathbb{E}_{(o,s) \sim p_{\text{data}}}[\log D(o, s)] + \mathbb{E}_{s \sim p_s, z \sim p_z}[\log(1 - D(G(s, z), s))]}
$$

| 符号 | 含义 |
|------|------|
| $p_{\text{data}}$ | 真实 (状态, 观测) 对的联合分布 |
| $p_s$ | 状态分布 |
| $p_z$ | 环境扰动分布 ($z \sim \mathcal{U}[-1, 1]$) |

**输入/输出规格**：

| 实验 | 状态维度 | z 维度 | 图像分辨率 | 训练数据量 |
|------|---------|--------|-----------|-----------|
| X-Plane 11 | $[p, \theta]$ 2 维 | 2 | 8×16 灰度 | 10,000 对 |
| CARLA | $[d, v]$ 2 维 | 4 | 32×32 灰度 | 400 对 |

**实现细节**：
- **初始化**：正交初始化
- **正则化**：判别器用谱归一化 (Spectral Normalization)
- **小数据集问题**：CARLA 仅 400 张图时 cGAN 出现模式坍塌 → 采用监督微调 MLP 替代

### 4.2 Step 2: MLP 蒸馏 + Lipschitz 正则化

**动机**：cGAN 生成器结构复杂，不利于形式化验证（α,β-CROWN 需要解析神经网络结构）。将其蒸馏为紧凑的 MLP。

**蒸馏损失**：

$$
\boxed{\mathcal{L}_{\text{distill}} = \|G(s, z) - g(s, z)\|_2^2 + \lambda_{\text{lip}} \cdot \mathcal{L}_{\text{lip}}}
$$

| 项 | 含义 | 作用 |
|----|------|------|
| $\|G(s, z) - g(s, z)\|_2^2$ | 输出一致性（ℓ2 损失） | 学生模型 $g$ 模仿教师 $G$ |
| $\mathcal{L}_{\text{lip}}$ | 谱 Lipschitz 损失 | 控制学生网络的全局 Lipschitz 常数 |
| $\lambda_{\text{lip}}$ | 正则化权重 | 平衡一致性与光滑性 |

**$\mathcal{L}_{\text{lip}}$ 的计算**：通过 **power iteration** 估计每层线性层的谱范数，结合激活函数和 BatchNorm 的 Lipschitz 因子。BatchNorm 层的贡献通过其缩放系数计算。这给出神经网络全局 Lipschitz 常数的**可计算近似**。

**为什么需要 Lipschitz 约束？**
- Theorem 3.1 的网格验证依赖所有组件（$B, f, \pi, g$）的 Lipschitz 连续性
- $g$ 的 Lipschitz 常数越小 → 网格常数 $K$ 越小 → 验证越容易通过

### 4.3 Step 3: PPO 预训练控制器

在蒸馏后的感知模型 $g(s, z)$ 上训练控制器 $\pi_\theta(a|o)$。策略基于从感知观测 $o = g(s, z)$ 导出的状态估计 $\hat{s}$ 做决策。

**PPO clipped surrogate objective**：

$$
\boxed{\mathcal{L}(\theta) = \hat{\mathbb{E}}_t\left[\min\Big(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\Big)\right]}
$$

| 符号 | 含义 |
|------|------|
| $r_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{\text{old}}}(a_t|o_t)}$ | 概率比 |
| $\hat{A}_t$ | 优势估计 (advantage estimator) |
| $\epsilon$ | clip 参数（PPO 标准值） |
| $\hat{\mathbb{E}}_t$ | 对时间步的经验期望 |

**实现**：基于 Stable-Baselines3 [26]。预训练得到初始控制器 $\pi_0$。

### 4.4 Step 4: 数据驱动的扰动估计 (Remark 1)

**过程**：

1. 对给定状态集 $\mathbb{S} = \{s_1, \ldots, s_{N_S}\}$，为每个 $s_i$ 生成 $N$ 个扰动样本
2. 计算基准环境下的下一状态：$s'_{i, z_0}$（使用 $z_0$）
3. 计算扰动环境下的下一状态：$s'_{i, z}$（使用随机 $z$）
4. 扰动：$\Delta s_i = s'_{i, z} - s'_{i, z_0}$
5. 统计所有状态和扰动样本上的 $\Delta s$ 分布 → 得到 $\mu$

$$
\boxed{\Delta s_i = s'_{i, z} - s'_{i, z_0}, \quad \mu = \text{EmpiricalDistribution}(\{\Delta s_i\})}
$$

### 4.5 Step 5: VCLS 组装

$$
\boxed{\text{VCLS} = g \circ \pi_0 \quad \text{即} \quad s_{t+1} = f(s_t, \pi_0(g(s_t, z_0))) + \Delta s}
$$

这是一个**完全可验证的闭环系统**：所有组件（$f$, $g$, $\pi_0$）都在低维空间中有解析形式。

---

## 五、组件二：SBC 引导的可证明安全控制器合成

### 5.1 神经 SBC 合成

用神经网络 $B(s)$ 近似候选随机障碍函数，最小化复合损失：

$$
\boxed{\mathcal{L}_L = \mathcal{L}_{\text{dec\_L}} + \lambda_R \cdot \mathcal{L}_{\text{region}} + \lambda_L \cdot \mathcal{L}_{\text{lip\_L}}}
$$

#### (a) 期望递减损失 $\mathcal{L}_{\text{dec\_L}}$（主导项）

$$
\boxed{\mathcal{L}_{\text{dec\_L}} = \mathbb{E}_{s_t}\Big[\max\big(0, \;\mathbb{E}[B(s_{t+1}) \mid s_t] - \gamma B(s_t) + \epsilon\big)\Big]}
$$

| 参数 | 含义 | 范围 |
|------|------|------|
| $\gamma$ | 衰减因子 | $\gamma \in (0, 1]$ |
| $\epsilon$ | 严格递减边际 | $\epsilon > 0$ |
| $\mathbb{E}[B(s_{t+1}) \mid s_t]$ | 条件期望 | 用 **Monte Carlo 采样**估计 |

**MC 采样实现**：从 $s_t$ 出发，采样 $M$ 条 $\Delta s$ 轨迹：
$$
\mathbb{E}[B(s_{t+1}) \mid s_t] \approx \frac{1}{M} \sum_{j=1}^{M} B(\tilde{F}(s_t, z_0, \Delta s^{(j)}))
$$

**损失的含义**：如果 $\mathbb{E}[B(s_{t+1})] > \gamma B(s_t) - \epsilon$（即期望递减不够），则被惩罚。注意这里用的是 $\gamma B(s_t)$ 而非 $B(s_t)$，允许 $\gamma < 1$ 时有一些松弛。

#### (b) 区域约束损失 $\mathcal{L}_{\text{region}}$

惩罚初始集 $\mathcal{S}_0$ 上 $B(s) > 1$（违反条件 ii）和不安全集 $X_u$ 上 $B(s) < 1/(1-p)$（违反条件 iii）的样本。

#### (c) Lipschitz 正则化 $\mathcal{L}_{\text{lip\_L}}$

约束 $B(s)$ 的 ℓ2 Lipschitz 常数，为 Theorem 3.1 的网格验证做准备。类似于 $g$ 的谱 Lipschitz 约束。

### 5.2 形式化验证

#### 条件 (i) 验证
通过神经网络架构设计自动满足（如最终层用 ReLU 或 Softplus → 输出 $\geq 0$）。

#### 条件 (ii) 和 (iii) 验证
使用 **α,β-CROWN** [34] 的区间传播：
- 在 $\mathcal{S}_0$ 上验证 $B(s) \leq 1$（条件 ii）
- 在 $X_u$ 上验证 $B(s) \geq 1/(1-p)$（条件 iii）
- α,β-CROWN 输出 $B(s)$ 在某个输入区域的上下界
- 上界 $\leq 1$ → 条件 (ii) 通过
- 下界 $\geq 1/(1-p)$ → 条件 (iii) 通过

#### 条件 (iv) 验证: Theorem 3.1 — Lipschitz 网格验证

**核心挑战**：条件 (iv) 需要在**连续状态空间**中验证，这在计算上等价于验证无穷多个点。

**核心技巧**：利用所有组件的 Lipschitz 连续性，将逐点验证转化为**有限网格点验证 + 容差**。

**定理陈述**：

设 $\tilde{F}(s, z_0, \Delta s) = f(s, \pi(g(s, z_0))) + \Delta s$。$B$ 的 Lipschitz 常数为 $L_B$，$f, \pi, g$ 的 Lipschitz 常数分别为 $L_f, L_\pi, L_g$。设网格间距为 $\tau$：$\|s - \tilde{s}\|_2 \leq \tau$。

定义**网格常数**：

$$
\boxed{K = \tau \cdot L_B \cdot \left(1 + L_f \sqrt{1 + (L_\pi L_g)^2}\right)}
$$

若存在网格点 $\tilde{s}$ 满足**强化条件**：

$$
B(\tilde{s}) - \mathbb{E}_{\Delta s \sim \mu}\big[B(\tilde{F}(\tilde{s}, z_0, \Delta s))\big] - K \geq \epsilon
$$

则该网格点 $\tau$-邻域内的**所有点** $s$ 都满足条件 (iv)：
$$
B(s) - \mathbb{E}_{\Delta s \sim \mu}\big[B(\tilde{F}(s, z_0, \Delta s))\big] \geq \epsilon
$$

**完整证明**：

**式 (1)** — $B(s)$ 的下界：
$$B(s) \geq B(\tilde{s}) - L_B \|s - \tilde{s}\|_2 \geq B(\tilde{s}) - \tau L_B$$

**式 (2)** — 期望项的上界：
$$\begin{aligned}
\mathbb{E}[B(\tilde{F}(s, z_0, \Delta s))] &= \mathbb{E}[B(f(s, \pi(g(s, z_0))) + \Delta s)] \\
&\leq \mathbb{E}[B(f(\tilde{s}, \pi(g(\tilde{s}, z_0))) + \Delta s)] \\
&\quad + L_B \|f(s, \pi(g(s, z_0))) - f(\tilde{s}, \pi(g(\tilde{s}, z_0)))\|_2 \\
&\leq \mathbb{E}[B(\tilde{F}(\tilde{s}, z_0, \Delta s))] + L_B L_f \sqrt{\|s-\tilde{s}\|_2^2 + \|\pi(g(s,z_0)) - \pi(g(\tilde{s},z_0))\|_2^2} \\
&\leq \mathbb{E}[B(\tilde{F}(\tilde{s}, z_0, \Delta s))] + \tau L_B L_f \sqrt{1 + (L_\pi L_g)^2}
\end{aligned}$$

**式 (3)** — 合并：
$$B(s) - \mathbb{E}[B(\tilde{F}(s, z_0, \Delta s))] \geq B(\tilde{s}) - \mathbb{E}[B(\tilde{F}(\tilde{s}, z_0, \Delta s))] - K$$

因此如果网格点满足 $B(\tilde{s}) - \mathbb{E}[\cdots] - K \geq \epsilon$，则邻域内所有点自动满足。

**$K$ 的构成解析**：

| 因子 | 来源 | 直观 |
|------|------|------|
| $\tau$ | 网格分辨率 | 网格越密，$K$ 越小 |
| $L_B$ | $B$ 的 Lipschitz 常数 | $B$ 越光滑，$K$ 越小 |
| $1 + L_f\sqrt{1+(L_\pi L_g)^2}$ | $f, \pi, g$ 的 Lipschitz 传播 | 系统整体越光滑，$K$ 越小 |

**验证流程**：
1. 在状态空间中铺设网格（间距 $\tau$）
2. 对每个网格点计算 $B(\tilde{s})$ 和 $\mathbb{E}[B(\tilde{F})]$（MC 采样）
3. 检查 $B(\tilde{s}) - \mathbb{E}[B(\tilde{F})] - K \geq \epsilon$
4. 若通不过 → 产生反例 → 加入反例集 → 用于训练

### 5.3 SBC 引导的控制器策略合成

控制器损失：

$$
\boxed{\mathcal{L}_P = \mathcal{L}_{\text{dec\_P}} + \lambda_P \cdot \mathcal{L}_{\text{lip\_P}} + \lambda_M \cdot \mathcal{L}_{\text{mse}}}
$$

| 损失项 | 作用 | 公式 |
|--------|------|------|
| $\mathcal{L}_{\text{dec\_P}}$ | 与 SBC 相同的鞅损失结构 | $\max(0, \mathbb{E}[B(s_{t+1})] - \gamma B(s_t) + \epsilon)$ |
| $\mathcal{L}_{\text{lip\_P}}$ | 控制器 Lipschitz ℓ2 正则化 | 约束 $\pi$ 的局部 Lipschitz 常数 |
| $\mathcal{L}_{\text{mse}}$ | 行为相似性约束 | $\mathbb{E}_o[\|\pi(o) - \pi_0(o)\|_2^2]$ |

**$\mathcal{L}_{\text{dec\_P}}$ 的机制**：当验证失败时，$\pi$ 收集反例并计算鞅损失 → 更新 $\pi$ 以选择使 SBC 值下降更多的动作。

**$\mathcal{L}_{\text{mse}}$ 的作用**：防止控制器为了通过 SBC 验证而完全偏离预训练基准（导致名义性能崩溃）。

### 5.4 交替优化策略 (Remark 2)

```
┌─────────────────────────────────────────────────────────┐
│                 交替优化循环                              │
│                                                         │
│  步骤 1: SBC 更新（固定 π）                               │
│    优化 L_L → 为当前策略提供更紧的概率安全界               │
│    频率：每轮                                            │
│          ↓                                              │
│  步骤 2: 策略更新（固定 B）                               │
│    优化 L_P → 控制器选择使 B 值下降更多的动作              │
│    频率：每 10 轮 SBC 更新后更新 1 次                      │
│          ↓                                              │
│  步骤 3: 扰动分布更新                                    │
│    每次策略更新后重新估计 Δs 分布                          │
│   （因为闭环动力学随 π 改变而改变）                        │
│          ↓                                              │
│  步骤 4: 形式化验证                                      │
│    检查 SBC 四个条件是否满足                              │
│    不满足 → 收集反例 → 回到步骤 1                         │
│    满足 → 输出 (π, SBC)                                 │
└─────────────────────────────────────────────────────────┘
```

**为什么交替比联合好？**

| 方案           | 结果                              | 原因                            |
| ------------ | ------------------------------- | ----------------------------- |
| 固定 π + SBC   | 84.9% (X-Plane) / 72.6% (CARLA) | 控制器不调整，SBC 只能适应固定的行为          |
| 联合更新 (Joint) | OT（不收敛）                         | SBC 和 π 同时变动 → 目标函数不稳定 → 无法收敛 |
| **交替优化**     | **92.1% / 94.2%**               | SBC 先稳定 → π 再适应 → 逐步收紧        |

---

## 六、总体算法

```
Algorithm 1: SafePVC
═══════════════════════════════════════════════════════════════
输入:  数据集 D = {(s_i, o_i)}, 系统动力学 f, 潜变量 z ~ U[-1,1]
       离散状态空间 S
输出:  SBC, 控制器网络 π

 1.  G, D_disc ← cGAN(D, z)                    // 训练 cGAN 感知模型
 2.  g ← Distill(G)                             // MLP 蒸馏 (带 Lipschitz)
 3.  π₀ ← InitController()                      // 初始化控制器网络
 4.  π₀ ← PPO(π₀, f, S)                        // RL 预训练 200k 步
 5.  π ← π₀
 6.  VCLS ← Concat(g, π₀)                       // 组装可验证闭环系统
 7.  SBC ← InitSBC()                            // 初始化 SBC 网络
 8.  Δs ← Distribution_Estimation(VCLS, f, z₀, z, S)  // 数据驱动扰动
 9.  iters ← 0
10.  while (SBC, VCLS, z₀, Δs, S) 不满足 Theorem 2.2 do
11.      收集验证过程中的反例 Cexs
12.      SBC ← Update_SBC(SBC, Loss_SBC(Cexs))        // 更新 SBC (每轮)
13.      if iters mod 10 == 0 then
14.          π ← Update_π(π, Loss_VCLS(Cexs))          // 更新 π (每10轮)
15.          Δs ← Distribution_Estimation(VCLS, f, z₀, z, S)  // 更新 Δs
16.      end if
17.      iters ← iters + 1
18.  end while
19.  return SBC, π
═══════════════════════════════════════════════════════════════
```

### 逐行解释

| 行 | 操作 | 详细说明 |
|----|------|----------|
| 1 | cGAN 训练 | 判别器+生成器对抗训练，谱归一化+正交初始化 |
| 2 | MLP 蒸馏 | cGAN 生成器 → 紧凑 MLP，加谱 Lipschitz 正则化 |
| 3-4 | PPO 预训练 | 在蒸馏模型上用 Stable-Baselines3 训练 200k 步 |
| 5-6 | VCLS 组装 | $s_{t+1} = F(s_t, z_0) + \Delta s$ 的可验证形式 |
| 7 | SBC 初始化 | 随机初始化神经 SBC 网络 |
| 8 | Δs 估计 | 从 $N_S$ 个状态和 $N$ 个扰动样本统计分布 |
| 9-18 | 主循环 | 反例驱动的交替优化直到验证通过 |
| 13 | 非对称频率 | $\pi$ 更新频率是 SBC 的 1/10，保证 SBC 先行稳定 |
| 15 | Δs 重估计 | 闭环动力学随 $\pi$ 改变，扰动分布必须更新 |

---

## 七、网络结构汇总

### 7.1 cGAN 生成器 G(s, z)

| 属性 | X-Plane 11 | CARLA |
|------|-----------|-------|
| 输入 | $[p, \theta] + z$ (2+2=4 维) | $[d, v] + z$ (2+4=6 维) |
| 架构 | 全连接网络 | 全连接网络 |
| 输出 | 8×16 灰度图 (128 维) | 32×32 灰度图 (1024 维) |
| 激活 | ReLU (隐藏), Tanh (输出) | 同 |
| 初始化 | 正交初始化 | 同 |

### 7.2 蒸馏 MLP g(s, z)

| 属性 | 值 |
|------|-----|
| 输入 | s + z（与 cGAN 相同） |
| 输出 | 与 cGAN 生成器相同维度 |
| 约束 | 谱 Lipschitz 正则化 |
| 训练 | 教师-学生蒸馏 |

### 7.3 控制器 π

| 属性 | 值 |
|------|-----|
| 输入 | 状态估计 $\hat{s}$（从 $g(s,z)$ 输出或隐式） |
| 输出 | 控制动作 $u$ |
| 架构 | 全连接 DNN |
| 训练 | PPO（200k 步）；安全微调（SBC 引导） |
| 超参数 | lr = $3 \times 10^{-4}$, batch size = 64 |

### 7.4 神经 SBC B(s)

| 属性 | 值 |
|------|-----|
| 输入 | 系统状态 $s$ |
| 输出 | 标量 $B(s) \geq 0$ |
| 输出约束 | 非负激活（如 Softplus/ReLU）保证条件 (i) |
| 正则化 | Lipschitz ℓ2 约束 |
| 验证工具 | α,β-CROWN（条件 ii, iii） + Theorem 3.1（条件 iv） |
| 学习率 | $1 \times 10^{-3}$ |

---

## 八、超参数汇总

| 参数 | 值 | 含义 |
|------|-----|------|
| **cGAN/感知模型** | | |
| $p$（潜变量维度） | 2 (X-Plane), 4 (CARLA) | 环境扰动 $z$ 的维度 |
| $z$ 分布 | $\mathcal{U}[-1, 1]$ | 均匀分布 |
| 图像分辨率 | 8×16 (X-Plane), 32×32 (CARLA) | 灰度图 |
| 训练数据量 | 10,000 (X-Plane), 400 (CARLA) | 状态-图像对 |
| **PPO 预训练** | | |
| learning rate | $3 \times 10^{-4}$ | |
| batch size | 64 | |
| training steps | 200,000 | |
| **SBC 训练** | | |
| SBC learning rate | $1 \times 10^{-3}$ | |
| 控制器 learning rate | $5 \times 10^{-3}$ | |
| $\gamma$ | $\in (0, 1]$ | 递减衰减因子 |
| $\epsilon$ | $> 0$ | 递减边际 |
| $\lambda_R$ | — | 区域约束权重 |
| $\lambda_L$ | — | Lipschitz 正则化权重 |
| $\lambda_P$ | — | 控制器 Lipschitz 权重 |
| $\lambda_M$ | — | 行为相似性约束权重 |
| **交替优化** | | |
| SBC 更新频率 | 每轮 | |
| $\pi$ 更新频率 | 每 10 轮 | |
| **验证** | | |
| $\tau$ | — | 网格间距 (Theorem 3.1) |
| $L_B, L_f, L_\pi, L_g$ | — | 各组件 Lipschitz 常数 |
| $N_S$ | — | 扰动估计的状态样本数 |
| $N$ | — | 每状态的扰动样本数 |
| $p$ | — | 目标安全概率 |
| **实验** | | |
| X-Plane 状态 | $p \in [-11, 11]$m, $\theta \in [-30, 30]^\circ$ | 横向偏差+航向偏差 |
| CARLA 状态 | $d \in [5, 16]$m, $v \in [0, 3]$m/s | 距离+速度 |
| Δt | 0.05s | 控制频率 |
| X-Plane 速度 | $v = 5$m/s | 恒速滑行 |
| X-Plane 轴距 | $L = 5$m | |

---

## 九、实验结果与消融分析

### 9.1 Table 1: 验证引导训练的有效性

| 方案                     | X-Plane 11           | CARLA                 | 说明              |
| ---------------------- | -------------------- | --------------------- | --------------- |
| 固定 π（不动控制器）            | 84.9% (40 iters)     | 72.6% (100 iters)     | SBC 只能适应固定策略    |
| 联合更新 (Joint) [37]      | OT (100 iters 内不收敛)  | OT                    | 同时更新两网络→不稳定     |
| **交替优化 (Alternating)** | **92.1%** (35 iters) | **94.2%** (100 iters) | SBC 先稳→π 再调→收紧了 |

**关键解读**：
- 联合训练的"OT"（Over Time，不收敛）表明同时更新 SBC 和 π 导致优化目标漂移
- 交替优化不仅收敛，而且需要的迭代次数更少（X-Plane: 35 vs 40）
- 交替比固定提高 7.2%（X-Plane）和 21.6%（CARLA）

### 9.2 SBC 可视化 (Fig. 3)

学习到的 SBC 在状态空间中呈现**分层的漏斗形状**：
- $\mathcal{S}_0$ 附近 $B(s) \approx 0\sim 1$（满足条件 ii）
- 安全区域 $B(s)$ 平滑递减（超鞅性质）
- 靠近 $X_u$ 时 $B(s)$ 急剧上升到 $\geq 1/(1-p)$（条件 iii）

### 9.3 鲁棒性实验 (Fig. 4)

在不同手动指定的扰动强度（均匀分布边界 = 状态空间 span 的百分比）下，安全概率下界随训练步数单调上升并收敛。扰动越大 → 收敛值越低 → 但仍能提供形式化下界。

---

## 十、关键数学原理深度解析

### 10.1 鞅理论的角色

SBC 条件 (iv) 的本质是：在安全区域内，$B(s_t)$ 构成的随机过程是一个**超鞅**。

**超鞅定义**：随机过程 $\{X_t\}$ 是超鞅，如果 $\mathbb{E}[X_{t+1} \mid \mathcal{F}_t] \leq X_t$。

SafePVC 中的对应：$\mathbb{E}[B(s_{t+1}) \mid s_t] \leq B(s_t) - \epsilon$（在条件 iv 的适用范围内）。

**直观**：B 值像一个**只走下坡路**的函数。从 1 开始，永远无法爬到 $1/(1-p)$ 的高度。

### 10.2 可选停止定理的角色

可选停止定理说：对于超鞅，$\mathbb{E}[X_\tau] \leq \mathbb{E}[X_0]$（在适当的技术条件下）。

应用到 SafePVC：
$$
\mathbb{E}[B(s_\tau)] \leq B(s_0) \leq 1
$$

但在 $X_u$ 上 $B \geq 1/(1-p)$，所以：
$$
\frac{1}{1-p} \cdot \mathbb{P}(\tau < \infty) \leq B(s_0) \leq 1
$$

解得 $\mathbb{P}(\tau < \infty) \leq 1-p$。

### 10.3 Lipschitz 网格验证的数学技巧

将无限维验证降为有限维的关键是不等式：
$$
B(s) - \mathbb{E}[B(\tilde{F}(s))] \geq B(\tilde{s}) - \mathbb{E}[B(\tilde{F}(\tilde{s}))] - K
$$

$K$ 量化了从网格点 $\tilde{s}$ 到邻域内任意 $s$ 的"信息衰减"。$K$ 越小（网格越密、网络越光滑）→ 验证越精确。

### 10.4 Markov 过程的角色

$\Delta s$ 与 $s$ 独立的假设（Assumption 1）保证了闭环系统构成**离散时间 Markov 过程**，这使得：
1. 条件期望 $\mathbb{E}[B(s_{t+1}) \mid s_t]$ 可以只用 $s_t$（不需要历史轨迹）
2. MC 采样估计期望值是有意义的（独立同分布样本）
3. 超鞅性质可以逐状态点验证（而非逐轨迹）

---

## 十一、论文独特之处与局限性

### 11.1 独特贡献

| # | 贡献 | 细节 |
|---|------|------|
| 1 | **首个 RL+SBC 联合框架** | 同时获得高性能（RL）和形式化保证（SBC） |
| 2 | **无限时间安全下界** | 唯一提供 $P(\text{safe}) \geq p$ 无限时间保证的视觉控制方法 |
| 3 | **数据驱动扰动建模** | Δs 直接从闭环仿真统计，无需人工指定 |
| 4 | **Lipschitz 网格验证** | 将连续空间逐点验证降为有限网格点验证 |
| 5 | **反例引导交替优化** | 比联合训练更稳定，比固定控制器更强 |
| 6 | **cGAN 蒸馏 + Lipschitz** | 在表达力和可验证性之间的关键平衡 |

### 11.2 局限性

1. **$f$ 必须已知**：系统动力学需完全已知，实际中难以满足
2. **$\Delta s$ 与 $s$ 独立**：强假设——现实中扰动可能与状态相关（如转弯时打滑更严重）
3. **低分辨率灰度图像**：8×16 和 32×32 分辨率限制了视觉信息量
4. **SBC Lipschitz 约束**：增加了训练的复杂性，可能限制表达力
5. **网格验证的维度爆炸**：$K \propto L_B L_f L_\pi L_g$，各组件 Lipschitz 常数叠乘
6. **反例引导迭代**：可能需要多轮收敛（X-Plane 35 轮，CARLA 100 轮）
7. **仅仿真验证**：没有物理平台或真实车辆实验
8. **cGAN 小数据模式坍塌**：CARLA 400 张数据不足以训练 cGAN → 改为监督 MLP

---

## 十二、与后续论文的关系

| 论文 | 关系 |
|------|------|
| **SPVT** (Paper 2) | SafePVC 的"全概率"与 SPVT 的"半概率"互补：一个无限时间强保证，一个 K 步可扩展 |
| **CBC-Latent** (Paper 3) | 共享"障碍证书"基因——SafePVC 在原始空间做 SBC，CBC-Latent 在潜空间做 CBC |
| **dCBF-Vision** (Paper 5) | SafePVC 的事后验证与 BarrierNet 的在线保证互补——可以事后用 SBC 认证在线 CBF 的长期安全性 |
