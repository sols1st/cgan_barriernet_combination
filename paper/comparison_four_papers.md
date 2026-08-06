# 四篇视觉安全控制论文详细比较

> 比较范围：SafePVC (Paper 1) · SPVT (Paper 2) · CBC-Latent (Paper 3) · dCBF-Vision (Paper 5)
>
> 共同主题：**视觉输入下的神经网络控制器安全保证**

---

## 〇、论文速览

| 论文                      | 全称                                                                             | 作者/机构                                         | 发表            | 一句话                         |
| ----------------------- | ------------------------------------------------------------------------------ | --------------------------------------------- | ------------- | --------------------------- |
| **Paper 1** SafePVC     | Provably Probabilistic Safe Controller Synthesis for Vision-Based NNCS         | Anonymous / DAC'26                            | 2026          | RL+SBC 鞅理论，无限时间全概率安全保证      |
| **Paper 2** SPVT        | Learning Vision-Based NN Controllers with Semi-Probabilistic Safety Guarantees | Ma, Wu, Sibai, Kantaros, Vorobeychik / WUSTL  | AAAI 2026     | 可达性+尾界，K 步半概率安全验证           |
| **Paper 3** CBC-Latent  | Safety Certification in the Latent Space using CBFs and World Models           | Anand, Kolathaya / IIT+IISc                   | arXiv 2025.07 | 世界模型潜空间+Transformer+CBC，半监督 |
| **Paper 5** dCBF-Vision | Differentiable CBFs for Vision-based End-to-End Autonomous Driving             | Xiao, Wang, Chahine, Amini, Hasani, Rus / MIT | arXiv 2022.03 | BarrierNet+CNN/LSTM，真实车辆部署  |

---

## 一、安全机制核心对比

### 1.1 安全哲学：四种不同的"安全"定义

```
Paper 1 (SafePVC):     "概率安全"  — P(safe) ≥ p，无限时间
                       工具：随机障碍证书 (SBC) + 鞅理论
                       
Paper 2 (SPVT):        "半概率安全" — 初始状态概率 + 环境最坏情况，有限K步
                       工具：K-可达性 + Chernoff-Hoeffding 尾界
                       
Paper 3 (CBC-Latent):  "经验安全"  — 学习鼓励安全，无形式化下界
                       工具：控制障碍证书 (CBC) + 潜空间世界模型
                       
Paper 5 (dCBF-Vision): "约束安全"  — CBF-QP 在线过滤，每步保证
                       工具：可微分 HOCBF + BarrierNet QP
```

### 1.2 安全机制详细对比

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **安全机制** | 随机障碍证书 SBC | K-可达性验证 | 控制障碍证书 CBC | 高阶控制障碍函数 HOCBF |
| **数学基础** | 鞅理论 (Supermartingale) | Chernoff-Hoeffding 分布无关尾界 | 正向不变性 (Forward Invariance) | CBF-QP + Lie 导数 |
| **安全保证类型** | **全概率**（初始+环境均为概率） | **半概率**（初始概率，环境最坏情况） | **无形式化保证**（仅鼓励安全） | **确定性约束**（每步 QP 保证） |
| **时间范围** | **无限时间** (infinite horizon) | **有限 K 步** (finite horizon) | 无明确保证范围 | **连续时间**（离散采样） |
| **安全下界** | $P(\text{safe}) \geq p$（可证明） | $\Pr \geq \frac{|V|}{N} - \sqrt{\frac{\log(2/\delta)}{2N}}$（可证明） | 无 | 若 QP 可解则安全（逐点保证） |
| **不安全集定义** | $X_u \subseteq \mathcal{S}$（Borel 可测） | 线性不等式（多面体） | $B(x) > 0$ 水平集 | $\psi_0(x) < 0$ 零下水平集 |
| **证书/证书函数** | $B(s) \in \mathbb{R}_{\geq 0}$ | 无证书（直接验证） | $B(z) \in \mathbb{R}$（可为负） | $\psi_i(x, z)$ 函数序列 |
| **证书训练方式** | 神经网络 + 四个损失条件 | N/A | 神经网络 + 安全/不安全分类 | QP 参数化，端到端梯度 |
| **证书是否可学习** | ✓（神经 SBC） | N/A | ✓（神经 CBC） | ✓（$p_i(z)$ 惩罚函数） |

---

## 二、感知模型与视觉处理

### 2.1 视觉处理管线

```
Paper 1 (SafePVC):
  真实图像 → [离线] cGAN训练 → MLP蒸馏(带Lipschitz正则化) → g(s,z) 作为可验证的感知代理
  部署时: 图像 → g⁻¹(估计s) → π(s) → u ... 或 图像 → π(o) → u

Paper 2 (SPVT):
  真实图像 → [离线] cGAN训练 → g(s,z) 作为感知代理
  部署时: 图像 → π(o) → u (生成器仅用于离线验证，不在线)
  
Paper 3 (CBC-Latent):
  真实图像 → DINO-v2(冻结) → z_t (384维) → Transformer d_θ 预测 z_{t+1}
  部署时: 图像 → DINO-v2 → z_t → π(z_t, p_t) → a_t

Paper 5 (dCBF-Vision):
  真实图像 → CNN(ResNet)+LSTM → z_t → 多级分支 → BarrierNet QP → u
  部署时: 图像 → 端到端前向传播 → QP求解 → u
```

### 2.2 感知模型详细对比

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **感知架构** | cGAN → MLP 蒸馏 | cGAN | DINO-v2 (ViT) + Transformer | CNN (ResNet) + LSTM |
| **输入** | 状态 $s$ + 潜变量 $z$ | 状态 $s$ + 潜变量 $z$ | RGB 图像 $o_t$（224×224） | RGB 图像 $o_t$（960×600） |
| **输出** | 灰度图像 (8×16 或 32×32) | 灰度图像 (8×16 或 64×64) | 384 维潜向量 $z_t$ | 特征向量 + 状态估计 + 控制 |
| **潜变量维度** | 2 (X-Plane), 4 (CARLA) | 10 (主实验), 2~64 (消融) | 384（DINO-v2 嵌入） | N/A（隐式在 CNN 特征中） |
| **训练数据量** | 10,000 (X-Plane), 400 (CARLA) | 20,000 (各场景), 400 (F1Tenth) | 50,000 转移 + ~250 轨迹 | ~400,000 图像 |
| **Lipschitz 约束** | ✓（蒸馏 MLP + SBC） | ✗ | ✗ | ✗（隐式在网络中） |
| **时序建模** | ✗（无记忆） | ✗（无记忆） | ✓（Transformer 因果注意力，H=3） | ✓（LSTM） |
| **生成器可微性** | ✗（事后蒸馏） | ✗（事后训练） | ✓（端到端，但 d_θ 冻结） | ✓（端到端） |
| **部署时需要感知模型** | 隐式（状态估计或端到端） | ✗（仅离线验证用） | ✓（DINO-v2 + Transformer） | ✓（CNN + LSTM） |

---

## 三、动力学模型

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **动力学来源** | 已知 $f$ | 已知 $f$ | **学习得到** $d_\theta$ | 已知 $f$（Frenet 坐标系） |
| **状态空间** | 原始物理状态 | 原始物理状态 | **潜空间**（384 维） | 原始物理状态（7 维 Frenet） |
| **状态维度** | 2 (X-Plane), 2 (CARLA) | 2~7 | 384 | 7 |
| **控制维度** | 1~2 | 1~2 | 1~2 | 2 (jerk + 转向加速度) |
| **离散/连续** | 离散时间 | 离散时间 | 离散时间 | **连续时间 ODE**（离散采样） |
| **随机性来源** | $\Delta s \sim \mu$（环境扰动） | $\omega \sim \Omega$（观测环境） | 随机动作探索 | 确定性（无显式噪声） |
| **是否假设动力学已知** | ✓（$f$ 已知） | ✓（$f$ 已知） | ✗（$d_\theta$ 学习，冻结） | ✓（$f$ 已知，Frenet 模型） |

---

## 四、训练方法论

### 4.1 训练目标与损失函数

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **训练范式** | 策略蒸馏（PPO预训练 + SBC精炼） | 梯度优化（预训练 anchor + 安全微调） | 半监督联合学习 | 策略蒸馏（NMPC→NN + dCBF） |
| **预训练** | PPO (RL) | 监督学习/PPO | ~250 轨迹（参考策略） | NMPC 离线最优 |
| **主损失** | $\mathcal{L}_L = \mathcal{L}_{\text{dec}} + \lambda_R \mathcal{L}_{\text{region}} + \lambda_L \mathcal{L}_{\text{lip}}$ | $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{perf}} + \lambda_2 \mathcal{L}_{\text{safety}}$ | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{barrier}} + \mathcal{L}_{\text{lie}} + \mathcal{L}_{\text{syn}} + \mathcal{L}_\pi$ | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{state}} + \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{control}}$ |
| **安全损失设计** | 鞅递减条件 $\max(0, \mathbb{E}[B(s_{t+1})] - \gamma B(s_t) + \epsilon)$ | 可达集变化率 $\frac{|\overline{\sigma}_K|+|\underline{\sigma}_K|}{K-1}$ | CBC 条件 $\max(0, B(z_{t+1}) - B(z_t))$ | 隐式在 QP 约束中（非损失项） |
| **性能保持** | $\mathcal{L}_{\text{mse}} = \|\pi - \pi_0\|^2$ | $\mathcal{L}_{\text{perf}}$ ($\ell_2$ 或 PPO) | $\mathcal{L}_\pi = \|\pi_\theta - \pi_{\text{user}}\|^2$ | $\mathcal{L}_{\text{control}} = \text{MSE}(\hat{u}, u^{\text{gt}})$ |
| **联合训练** | ✗（交替优化） | ✗（分阶段） | ✓（B 和 π 同时训练） | ✓（端到端反向传播） |
| **交替/迭代设计** | SBC 每轮更新，π 每 10 轮更新 | 自适应数据集 + 课程学习 | B 早停，π 继续训练 | 端到端梯度通过 OptNet KKT |

### 4.2 训练数据策略

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **数据来源** | 仿真采集（离线） | 仿真采集 （离线） | 随机动作 + 参考策略 | 真实世界 + NMPC + VISTA |
| **数据量** | 10,000 / 400 对 | 20,000 / 400 对 | 50,000 转移 + ~250 轨迹 | ~400,000 图像 |
| **自适应数据** | ✗（随机采样） | ✓（优先队列 $\mathbb{S}_A$ + 加权采样） | ✗（随机初始化轨迹） | ✗（NMPC 轨迹） |
| **课程学习** | ✗ | ✓（逐步增加 $K$） | ✗ | ✗ |
| **标注需求** | 状态-图像对 | 状态-图像对 | 安全/不安全区域标注 | 状态-图像-控制三元组 |

---

## 五、验证与保证机制

### 5.1 验证方法对比

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **验证工具** | α,β-CROWN (条件 ii,iii) + Lipschitz网格 (条件 iv) | α,β-CROWN (迭代 K 步可达性) | 无形式化验证 | 无事后验证（在线 QP 保证） |
| **验证对象** | SBC 四个条件 | K 步可达集安全性 | N/A | QP 约束满足 |
| **验证时机** | 训练中（反例引导） | 训练后（离线） | N/A | 每步运行时 |
| **反例引导** | ✓（cex 交替精炼） | ✗ | ✗ | ✗ |
| **可扩展性瓶颈** | Lipschitz 网格密度 + SBC 训练 | 验证样本数 $N$ × K 步 α,β-CROWN | N/A | QP 求解速度（<10ms，可接受） |
| **保守性来源** | SBC 设计 + 网格过近似 | $\epsilon$-ball 过近似 + 尾界惩罚项 | 潜动力学预测误差 | CBF 约束收紧（$p_i(z)$ 缓解） |

### 5.2 安全保证的量化对比

| 论文 | 保证强度 | 保证是否有下界 | 下界形式 |
|------|---------|---------------|----------|
| SafePVC | ★★★★★ 最强 | ✓ | $P(\text{safe}) \geq p$，无限时间 |
| SPVT | ★★★★ 强 | ✓ | $P_K(\text{safe}) \geq \hat{\alpha} - \epsilon(N,\delta)$，K 步 |
| dCBF-Vision | ★★★ 中 | ✗ | 逐点：若 QP 可解则安全 |
| CBC-Latent | ★ 弱 | ✗ | 无形式化保证 |

---

## 六、部署与实验

### 6.1 实验环境

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **仿真环境** | X-Plane 11, CARLA | X-Plane 11, CARLA, AirSim | 倒立摆, Dubin's Car | VISTA (数据驱动仿真) |
| **物理平台** | ✗ | F1Tenth (1:10 小车) | ✗ | **Lexus RX 450H** (全尺寸真车) |
| **真实车辆** | ✗ | ✗ | ✗ | ✓ (雪地+冰面+AR避障) |
| **图像分辨率** | 8×16, 32×32 灰度 | 8×16, 64×64 灰度 | 224×224 RGB | 960×600 RGB |
| **任务类型** | 轨迹跟踪, 紧急制动 | 飞机着陆, 车道跟随, 无人机 | 倒立摆稳定, 避障 | 车道保持, 避障 |

### 6.2 关键实验结果

| 论文 | 核心指标 | 核心数字 |
|------|---------|----------|
| SafePVC | 可验证安全概率下界 | X-Plane: **92.1%** (交替优化) vs 84.9% (固定) |
| SPVT | 安全概率下界 + 经验性能 | 所有场景中**最高的安全概率下界**；无人机场景 VSRL **完全失败** |
| CBC-Latent | 安全区域分离 + 轨迹可视化 | 倒立摆从非安全→安全区域收敛；Dubin's Car 成功避障 |
| dCBF-Vision | 碰撞率 + 最小间隙 | w/ dCBF: 28%→**3%** (GT state)；真实车辆雪地急转弯成功 |

---

## 七、环境不确定性处理

这是四篇论文最核心的差异维度之一。

```
环境不确定性的三种处理哲学：

  SafePVC:          SPVT:               dCBF-Vision:       CBC-Latent:
  全概率             半概率               隐式处理            学习适应
  
  Δs ∼ μ           ω ∈ Ω 最坏情况        p_i(z) 学习自适应    d_θ 学习动力学
  (数据驱动分布)     (∀ω, 不确定集)       (惩罚函数放松收紧)    (Transformer预测)
  
  初始状态: 全空间    初始状态: 采样+尾界   初始状态: 训练覆盖    初始状态: 训练覆盖
  环境: 概率分布     环境: 最坏情况       环境: 在线适应       环境: 隐式学习
```

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **初始状态处理** | $\forall s_0 \in \mathcal{S}_0$（全空间） | $N$ 个 i.i.d. 样本 + 尾界 | 随机初始化 ~250 轨迹 | 训练数据覆盖的分布 |
| **环境扰动建模** | $\Delta s \sim \mu$（数据驱动，与 $s$ 独立） | $\omega \in \Omega$（最坏情况，$\forall \omega$） | 隐式（潜动力学 $d_\theta$ 学习） | 隐式（$p_i(z)$ 适应 + VISTA 多样化） |
| **观测不确定性** | cGAN $g(s,z)$ + Lipschitz MLP 蒸馏 | cGAN $g(s,z)$ + $\epsilon$-ball 过近似 | DINO-v2 编码（确定性） | CNN+LSTM（隐式不确定性） |
| **分布假设** | $\Delta s$ 有界 + 与 $s$ 独立 | $D$ 可采样 i.i.d.（分布无关） | 训练覆盖足够 | 训练数据多样性足够 |
| **不确定性的保守度** | 中等（$\mu$ 实际分布） | **高**（$\forall \omega$ 最坏情况） | **低**（依赖数据覆盖） | **可调**（$p_i(z)$ 自适应） |

---

## 八、数学理论深度对比

### 8.1 核心数学工具

| 论文 | 核心数学 | 关键定理 |
|------|---------|----------|
| SafePVC | **鞅理论**（Supermartingale）+ **可选停止定理** + Hoeffding 不等式 + Lipschitz 连续性 | Theorem 2.2 (SBC 四条件) · Theorem 3.1 (网格验证) |
| SPVT | **可达性分析** + **Chernoff-Hoeffding 分布无关尾界** + **归纳法** | Theorem 1 (可达集包含) · Theorem 2 (概率下界) |
| CBC-Latent | **正向不变性** (Forward Invariance) + **Transformer 自回归预测** + Lie 正则化 | Lemma 1 (CBC 安全性) |
| dCBF-Vision | **Lie 导数** + **HOCBF 序列构造** + **KKT 条件** (可微 QP) | Theorem 1 (HOCBF 安全性) · Def 4 (HOCBF) |

### 8.2 Lipschitz 连续性的角色

| 论文 | 是否使用 Lipschitz | 用途 |
|------|-------------------|------|
| SafePVC | **核心依赖** | 网格验证 (Theorem 3.1)：$K = \tau L_B(1+L_f\sqrt{1+(L_\pi L_g)^2})$，SBC + 控制器都要约束 |
| SPVT | 假设系统 Lipschitz | Assumption 1 验证 + 可达性计算 |
| CBC-Latent | 未显式使用 | N/A |
| dCBF-Vision | 隐含在 CBF 中 | $f, g$ 局部 Lipschitz（标准 CBF 前提），$p_i(z)$ 连续可微 |

### 8.3 从理论到实践的"最后一公里"

```
SafePVC:   理论：鞅 → 可选停止 → P(safe)≥p
           实践：SBC网络训练 + 网格验证 + 反例精炼 → 控制器满足下界

SPVT:      理论：ϵ-覆盖 + 归纳 → 可达集包含 + Hoeffding → 概率下界
           实践：cGAN验证 + α,β-CROWN K步验证 + 尾界计算 → 控制器满足下界

CBC-Latent: 理论：正向不变性 B(f(x,u))≤B(x) → 安全
           实践：潜空间B网络 + Transformer预测 → 鼓励安全(无下界)

dCBF-Vision: 理论：HOCBF ψ_i序列 + QP → 前向不变性
           实践：端到端训练 + OptNet QP求解 → 每步安全过滤
```

---

## 九、可扩展性分析

| 维度 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| **高维图像** | 差（需要蒸馏到低维） | 中（灰度+低分辨率） | **好**（DINO-v2 预训练） | 中（需要大训练集） |
| **复杂动力学** | 差（需要已知 $f$） | 中（需要已知 $f$，低维） | **好**（学习得到） | 中（Frenet模型限于车辆） |
| **多障碍物** | 未处理 | 未处理（单场景） | 简单（Dubin's Car 单障碍） | ✓（大圆盘覆盖策略） |
| **实时性** | 好（轻量MLP） | 好（离线验证） | 中（Transformer推理） | **好**（QP<10ms，总<33ms） |
| **新环境适应** | 差（需重新训练 cGAN + SBC） | 中（需重新训练 cGAN） | **好**（DINO-v2 零样本泛化） | 差（需大量新标注数据） |
| **安全保证可移植性** | 差（SBC 与 f,π 耦合） | 中（验证与 π,g 耦合） | 无形式化保证 | 中（QP 在线，无需重验证） |

---

## 十、理论流派分类

四篇论文代表了视觉安全控制领域的三种理论流派（外加一个实用主义分支）：

### 流派 A：障碍证书派 (Barrier Certificate School)
**SafePVC + CBC-Latent**

```
共同基因：学习一个"障碍函数"B(·)，用 B 的值来定义安全/不安全
分歧点：
  SafePVC: B(s) ≥ 0（非负），在原始状态空间，鞅理论，概率性
  CBC-Latent: B(z) 可为负（零水平集分离），在潜空间，正向不变性，确定性
```

### 流派 B：可达性验证派 (Reachability Verification School)
**SPVT**

```
核心：不学习证书，直接计算可达集并验证安全性
创新：半概率（分布无关尾界），ϵ-覆盖假设，可微分安全代理损失
哲学：与其相信一个学习的证书，不如直接检查可达状态
```

### 流派 C：在线安全过滤派 (Online Safety Filter School)
**dCBF-Vision**

```
核心：不事后验证，也不学习证书——而是让安全约束成为控制器的一部分
创新：可微分 QP 层，HOCBF 参数化，多级可解释架构
哲学：安全问题应该在控制输出端解决，而非在训练后验证
```

---

## 十一、相互引用与演进关系

```
                    ┌──────────────────────────────┐
                    │  CBF 理论 (Ames et al. 2014) │
                    └──────────────┬───────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
  ┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
  │ 障碍证书 (BC) │     │ HOCBF (Xiao 2019)│     │ 可达性分析        │
  │ Prajna 2004   │     │                 │     │                   │
  └───────┬───────┘     └────────┬────────┘     └────────┬──────────┘
          │                      │                       │
   ┌──────┴──────┐        ┌──────┴──────┐         ┌──────┴──────┐
   ▼             ▼        ▼             ▼         ▼             ▼
SafePVC      CBC-Latent  dCBF-Vision   SPVT       VSRL
(2026)       (2025)      (2022)        (2026)     (2024)
  
  鞅理论        潜空间      BarrierNet   半概率      全空间验证
  全概率        Transformer 真实车辆     K步保证     (SPVT前身)
  无限时间      半监督      端到端       尾界
```

**演进逻辑**：
- SafePVC 和 CBC-Latent 共享障碍证书基因，但 SafePVC 走"理论严格"路线（鞅+概率），CBC-Latent 走"实用主义"路线（潜空间+世界模型）
- dCBF-Vision 走"在线过滤"路线，BarrierNet 是 CBF-QP 的可微分神经网络化
- SPVT 走"验证"路线，用可达性替代证书，用半概率平衡可扩展性和保证强度

---

## 十二、互补性分析

### 12.1 四篇论文的理论互补

| 问题 | SafePVC 怎么解决 | SPVT 怎么解决 | CBC-Latent 怎么解决 | dCBF-Vision 怎么解决 |
|------|-----------------|--------------|-------------------|---------------------|
| 视觉→状态 | cGAN+MLP 蒸馏 | cGAN 生成器 | DINO-v2 编码 | CNN+LSTM 隐式 |
| 动力学 | 假设已知 $f$ | 假设已知 $f$ | Transformer 学习 | 假设已知 $f$ (Frenet) |
| 环境不确定性 | $\Delta s$ 概率分布 | $\omega$ 最坏情况 | 潜在动力学学习 | $p_i(z)$ 自适应 |
| 安全下界 | ✓ 无限时间 | ✓ K 步 | ✗ | ✗ |
| 实时安全 | 事后保证 | 事后保证 | 无形式化保证 | ✓ 在线 QP 过滤 |

### 12.2 理想的"超级系统"

结合四篇论文的优势，一个理想的视觉安全控制系统可能长这样：

```
视觉输入 → [CBC-Latent的DINO-v2编码器] → 潜空间状态
  潜空间状态 → [SafePVC的SBC] → 长期概率安全评估
  潜空间状态 → [SPVT的SPV] → K步可达性验证
  潜空间状态 → [dCBF-Vision的BarrierNet] → 实时安全控制过滤
```

---

## 十三、关键参数与超参数汇总

| 参数 | SafePVC | SPVT | CBC-Latent | dCBF-Vision |
|------|---------|------|------------|-------------|
| 安全概率阈值 | $p$ | N/A（尾界自动推导） | N/A | N/A |
| 置信度 | N/A（鞅） | $\delta = 0.05$ | N/A | N/A |
| 验证视野 | 无限 | $K$（可变，课程学习） | 无限（理想） | 连续时间 |
| 递减系数 | $\gamma \in (0,1]$ | N/A | $\alpha$ (Lie loss) | $\alpha_i(\cdot)$ (K类函数) |
| 递减边际 | $\epsilon > 0$ | N/A | N/A | N/A |
| 覆盖误差 | N/A | $\epsilon$ (Assumption 1) | N/A | N/A |
| 安全阈值 | N/A | $\beta$ (车道边界) | N/A | $d_{\text{lf}}$ (车道), $r_D$ (圆盘) |
| CBF 惩罚函数 | N/A | N/A | N/A | $p_i(z) > 0$, $m$ 个 |
| QP 代价矩阵 | N/A | N/A | N/A | $H(z|\theta_h)$, $F(z|\theta_f)$ |
| 温度参数 | N/A | $\alpha$ (加权采样) | N/A | N/A |
| 采样比例 | N/A | $p$ (来自 $\mathbb{S}_A$) | N/A | N/A |
| 批次大小 | 64 | 128 | 未明确 | 未明确 |
| 学习率 (SBC/B) | $1 \times 10^{-3}$ | N/A | 未明确 | N/A |
| 学习率 (控制器) | $5 \times 10^{-3}$ | $2\times 10^{-4}$ (安全训练) | 未明确 | 未明确 |
| 训练迭代 | 40~100 (SBC), 35~100 (控制器) | 100 epochs | 收敛为止 | 200 epochs |
| Lipschitz 常数 | $L_B, L_f, L_\pi, L_g$ (约束) | N/A | N/A | N/A |
| 网格间距 | $\tau$ (Theorem 3.1) | N/A | N/A | N/A |
| 潜变量分布 | $z \sim \mathcal{U}[-1,1]$ | $z \sim \mathcal{U}[-1,1]$ | N/A | N/A |
| 上下文长度 | N/A | N/A | $H=3$ | N/A |
| 预测视野 | N/A | N/A | 3 | N/A |
| 图像分辨率 | 8×16, 32×32 | 8×16, 64×64 | 224×224 | 960×600 |
| 数据集大小 | 10k, 400 | 20k, 400, 20k | 50k 转移 | ~400k |
| 实验平台 | X-Plane, CARLA | X-Plane, CARLA, AirSim, F1Tenth | 倒立摆, Dubin's Car | VISTA + Lexus RX 450H |

---

## 十四、选择指南：什么时候用哪篇？

| 场景 | 推荐论文 | 理由 |
|------|---------|------|
| 需要**严格无限时间安全下界** | SafePVC | 唯一提供 $P(\text{safe}) \geq p$ 无限时间保证的 |
| 需要**可扩展的 K 步验证** + 物理实验 | SPVT | 半概率框架 + F1Tenth 实车 |
| **没有动力学模型** + 视觉输入 | CBC-Latent | 唯一不假设已知 $f$ 的；Transformer 学习动力学 |
| 需要**真实车辆部署** + 在线安全 | dCBF-Vision | 唯一部署到全尺寸真车 (Lexus) 的 |
| **数据极少**（几百张图） | SafePVC 或 SPVT | CARLA 400 张, F1Tenth 400 张 |
| **高维复杂视觉** | CBC-Latent | DINO-v2 预训练 + 224×224 RGB |
| **多障碍物场景** | dCBF-Vision | 大圆盘覆盖策略 |
| **需要在线实时安全过滤** | dCBF-Vision | BarrierNet QP < 10ms |
| **需要端到端可微训练** | dCBF-Vision 或 CBC-Latent | 梯度直通 |
| **追求理论深度** | SafePVC | 鞅理论 + 可选停止定理 |

---

## 十五、总结

四篇论文从不同角度解决了"视觉输入下如何保证神经网络控制器安全"这一核心问题：

- **SafePVC** 走"理论严格"路线，用鞅理论提供最强的安全保证，但假设也最多（动力学已知、扰动独立）；**适合对安全有极严格要求的场景。**

- **SPVT** 走"工程平衡"路线，半概率设计既有形式化下界又可扩展；**适合需要验证但又能接受 K 步保证的场景。**

- **CBC-Latent** 走"数据驱动"路线，用现代视觉 Transformer 和世界模型降低对动力学的依赖；**适合没有动力学模型但有视觉数据的场景。**但牺牲了形式化保证。

- **dCBF-Vision** 走"系统集成"路线，将安全约束嵌入可微分 QP 层，端到端训练并部署到真实车辆；**适合需要实际部署的工程场景。**安全保证弱于 SafePVC 和 SPVT，但实用性强。
