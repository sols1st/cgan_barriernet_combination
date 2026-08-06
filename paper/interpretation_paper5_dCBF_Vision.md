# Paper 5 详细解读：dCBF-Vision — 面向视觉端到端自动驾驶的可微分控制障碍函数

> **论文标题**: Differentiable Control Barrier Functions for Vision-based End-to-End Autonomous Driving
>
> **作者**: Wei Xiao\*, Tsun-Hsuan Wang\*, Makram Chahine, Alexander Amini, Ramin Hasani, Daniela Rus (\* equal contributions)
>
> **机构**: MIT CSAIL
>
> **发表**: arXiv:2203.02401v1 (Mar 2022)
>
> **代码**: BarrierNet/Driving/ (基于 PyTorch Lightning + VISTA 仿真器)
>
> **核心贡献**: 将 BarrierNet 扩展到视觉端到端自动驾驶，构建多级可解释架构，融合 CNN/LSTM 视觉编码、HOCBF 安全约束和可微分 QP，在 sim-to-real 环境和全尺寸真实自动驾驶汽车（Lexus RX 450H）上验证了安全车道保持和避障。

---

## 〇、代码结构与参数速查

### 0.1 关键文件

| 文件 | 内容 |
|------|------|
| `models/barrier_net.py` | **主模型**：CNN+LSTM+dCBF+BarrierNet QP，所有网络连接 |
| `models/cnn_lstm.py` | 基线模型：CNN+LSTM+Q_MLP（无 BarrierNet） |
| `models/state_net.py` | 独立的状态估计网络（消融/替代方案） |
| `models/base.py` | 基类：损失函数 `_compute_loss()`、训练/验证 step |
| `models/utils.py` | `build_cnn()`、`build_mlp()`、QP 求解器 |
| `task_cbf.py` | VISTA 仿真环境：多车任务、碰撞检测、终止条件 |
| `run.py` | 训练入口 |

### 0.2 核心超参数默认值

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `cnn_params` | `[[3,24,5,2,2],[24,36,5,2,2],[36,48,3,2,1],[48,64,3,1,1],[64,64,3,1,1]]` | CNN 五层配置 `[in,out,kernel,stride,pad]` |
| `cnn_dropout` | 0.3 | CNN dropout |
| `lstm_size` | 64 | LSTM 隐藏维度 |
| `q_mlp_params` | `[32, 32, 2]` | Q_MLP 隐层配置，输出 [v̂, δ̂] |
| `p_mlp_params` | `[32, 32, 2]` | P_MLP 隐层配置，输出 [p₁, p₂] |
| `model_type` | `'deri'` | 导数模式（v̂,δ̂→导数→参考控制→QP→u） |
| `lr` | 1e-3 | Adam 学习率 |
| `batch_size` | 由 DataModule 指定 | — |
| `state_tol` | 0.05 | 状态估计 clamp 容差（±5%） |
| `lf_cbf_threshold` | 2.0 | 车道保持 CBF 阈值 (m) |
| `disk_radius` | 7.9 | 障碍物覆盖圆盘半径 (m) |

---

## 一、完整网络架构：从像素到安全控制

### 1.1 总体架构 —— 逐层连接

基于 `barrier_net.py` 的 `forward()` 方法，以下是从输入到输出的**精确数据流**：

```
═══════════════════════════════════════════════════════════════════════════════
                  dCBF-Vision 完整网络前向传播（基于代码）
═══════════════════════════════════════════════════════════════════════════════

输入 batch:
  img_seq  : (B, T, 3, H, W)     ← 图像序列（B=batch, T=时序长度）
  state_seq: (B, T, 7)           ← 真值状态 [s, d, μ, v, a, δ, κ]
  obs_seq  : (B, T, 2)           ← 真值障碍物 [Δs, d_obs]
  ctrl_seq : (B, T, 4)           ← 真值控制 [v_gt, δ_gt, a_gt, ω_gt]

═══════════════════════════════════════════════════════════════════════════════
【模块 A】CNN 空间编码器
═══════════════════════════════════════════════════════════════════════════════

  img_seq_fl = img_seq.view(B*T, 3, H, W)    ← 合并 B,T 维度

  Layer 1: Conv2d(3,   24, kernel=5, stride=2, pad=2) → BN → ReLU → Dropout(0.3)
  Layer 2: Conv2d(24,  36, kernel=5, stride=2, pad=2) → BN → ReLU → Dropout(0.3)
  Layer 3: Conv2d(36,  48, kernel=3, stride=2, pad=1) → BN → ReLU → Dropout(0.3)
  Layer 4: Conv2d(48,  64, kernel=3, stride=1, pad=1) → BN → ReLU → Dropout(0.3)
  Layer 5: Conv2d(64,  64, kernel=3, stride=1, pad=1) → BN → ReLU → Dropout(0.3)

  输出形状: (B*T, 64, H', W')

  z = z.max(-1)[0].max(-1)[0]     ← 对 H,W 维做 max pooling → (B*T, 64)
  cnn_feat = z                     ← 保存 CNN 特征（供 StateNet 可选使用）
  z = z.view(B, T, 64)            ← 恢复时序维度 → (B, T, 64)

═══════════════════════════════════════════════════════════════════════════════
【模块 B】LSTM 时序编码器
═══════════════════════════════════════════════════════════════════════════════

  z, rnn_state = LSTM(z, rnn_state)
    输入: (B, T, 64)
    LSTM hidden_size = 64, num_layers = 1
    输出: (B, T, 64)

  z = z.reshape(B*T, 64)          ← 展平为 (B*T, 64)

═══════════════════════════════════════════════════════════════════════════════
【模块 C】双分支 MLP（从同一个 LSTM 特征 z 分出）
═══════════════════════════════════════════════════════════════════════════════

  分支 1: Q_MLP (参考控制预测 — Level 2)
  ─────────────────────────────────────
    q = self._q_mlp(z)              ← z: (B*T, 64)
    MLP: 64 → 32 → ReLU → Dropout → 32 → ReLU → Dropout → 2
    输出: q = [v̂, δ̂]  (B*T, 2)    ← 预测的速度和转向角

  分支 2: P_MLP (惩罚函数预测)
  ─────────────────────────────
    p = self._p_mlp(z) * 4          ← z: (B*T, 64), ×4 是放大因子
    MLP: 64 → 32 → ReLU → Dropout → 32 → ReLU → Dropout → 2 → Sigmoid
    输出: p = [p₁, p₂]  (B*T, 2)   ← 每个避障约束有 2 个惩罚函数（m=2）

    若 p_lower_bound != None: p = clamp(p, min=p_lower_bound)

═══════════════════════════════════════════════════════════════════════════════
【模块 D】导数层 (Level 3) — 仅在 model_type='deri' 时
═══════════════════════════════════════════════════════════════════════════════

  v_delta = q                     ← 保存 [v̂, δ̂] 用于输出和损失

  q_for_qp = -((v_delta - state_seq[:, 3:5]) / 0.1)
    展开: q_for_qp[0] = -(v̂ - v_gt) / 0.1     ← 速度跟踪误差的负值（÷0.1 缩放）
          q_for_qp[1] = -(δ̂ - δ_gt) / 0.1     ← 转向跟踪误差的负值

  解读: q_for_qp 是"参考控制的负跟踪误差"。QP 最小化 ½u² + q·u
        如果 v̂ < v_gt → q > 0 → QP 倾向负 u（加速）
        如果 v̂ > v_gt → q < 0 → QP 倾向正 u（减速）
        → QP 层的参考控制作用是"让实际控制逼近 v̂,δ̂ 所隐含的动态"

═══════════════════════════════════════════════════════════════════════════════
【模块 E】StateNet (Level 1) — 可选，当 use_state_net=True
═══════════════════════════════════════════════════════════════════════════════

  state_net_in = z (若 use_lstm_in_state_net) 或 cnn_feat (若不用 LSTM)

  独立的 MLP 分支（每个状态变量一个 MLP，ELU 激活）:

  ds    = self._ds_mlp(state_net_in)     MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1
  dd    = self._dd_mlp(state_net_in)     MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1
  d     = self._d_mlp(state_net_in)      MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1
  mu    = self._mu_mlp(state_net_in)     MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1
  kappa = self._kappa_mlp(state_net_in)  MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1
  obs_d = self._obs_d_mlp(state_net_in)  MLP: 64→32→ReLU→Dropout→32→ReLU→Dropout→1

  稳定性机制（训练时）:
    用 .detach() 切断梯度
    用 clamp(., gt*(1-tol), gt*(1+tol)) 限制在真值 ±5% 范围内
    非 detach 版本保留用于损失计算

  若不用 StateNet: 直接使用 GT 状态 → d=gt_d, mu=gt_mu, ds=gt_ds, dd=gt_dd

═══════════════════════════════════════════════════════════════════════════════
【模块 F】车辆运动学计算
═══════════════════════════════════════════════════════════════════════════════

  车辆参数: lrf = 0.5 (= lr/(lr+lf)), lr = 2.78 (轴距)

  beta = atan(lrf * tan(δ))
  cos_mu_beta = cos(μ + β)
  sin_mu_beta = sin(μ + β)
  mu_dot = v/lr * sin(beta) - κ * v * cos_mu_beta / (1 - d*κ)

═══════════════════════════════════════════════════════════════════════════════
【模块 G】避障 dCBF 约束构造 (相对度 m=2)
═══════════════════════════════════════════════════════════════════════════════

  圆盘半径: r_D = 7.9m

  barrier      = ds² + dd² - 7.9²                            ← ψ₀ = b(x)
  barrier_dot  = 2*ds*v*cos_mu_beta/(1-dκ) + 2*dd*v*sin_mu_beta  ← ψ̇₀
  Lf2b         = ...                                          ← L_f² b(x), 完整推导见下文
  LgLfbu1      = 2*ds*cos_mu_beta/(1-dκ) + 2*dd*sin_mu_beta  ← L_gL_f b(x) 第 1 列
  LgLfbu2      = (...) * lrf/cos²δ/(1+(lrf*tanδ)²)          ← L_gL_f b(x) 第 2 列

  G_obs = -[LgLfbu1, LgLfbu2]  shape: (B*T, 1, 2)
  h_obs = Lf2b + (p₁+p₂)*barrier_dot + p₁*p₂*barrier    ← dCBF: α₁=p₁+p₂, α₂=p₁·p₂

═══════════════════════════════════════════════════════════════════════════════
【模块 H】车道保持 dCBF 约束构造 (若 use_lane_keeping_CBFs=True)
═══════════════════════════════════════════════════════════════════════════════

  左边界: barrier = 2.0 - d                           ← b_left = d_lf - d ≥ 0
  右边界: barrier = d + 2.0                           ← b_right = d_lf + d ≥ 0

  车道保持使用固定 α₁=2, α₂=1（非可学习）:
    h = Lf2b + 2*barrier_dot + 1*barrier

  最终 G = concat([G_obs, G_left, G_right], dim=1)  shape: (B*T, 3, 2)
  最终 h = concat([h_obs, h_left, h_right], dim=1)  shape: (B*T, 3)

═══════════════════════════════════════════════════════════════════════════════
【模块 I】BarrierNet QP 求解
═══════════════════════════════════════════════════════════════════════════════

  Q = eye(2) 展开到 (B*T, 2, 2)     ← 控制代价矩阵，对 jerk 和 steer 等权重
  q = q_for_qp                       ← 参考控制向量

  QP:
    min_u  ½ u^T Q u + q^T u
    s.t.   G u ≤ h                   ← 注意代码中是 G u ≤ h（非 ≥）
           (无等式约束, 无控制边界 — 训练时放松)

  求解器: QPFunction(PDIPM_BATCHED)  ← qpth 库的批量原对偶内点法
  输出: x = [u_jerk, u_steer]  (B*T, 2)

═══════════════════════════════════════════════════════════════════════════════
【模块 J】输出组装
═══════════════════════════════════════════════════════════════════════════════

  model_type='deri':
    out = concat([v_delta, x], dim=1)      ← [v̂, δ̂, u_jerk, u_steer]  4维
    若 use_state_net:
      out = concat([out, ds, dd, mu, ...]) ← 加上状态估计

  out = out.view(B, T, -1)                ← 恢复时序维度

═══════════════════════════════════════════════════════════════════════════════
```

### 1.2 连接关系总结

```
                      ┌──────────────────────────────────────────────┐
                      │              一张图看懂所有连接                │
                      └──────────────────────────────────────────────┘

  img_seq (B,T,3,H,W)
       │
       ▼
  ┌─────────┐
  │   CNN   │ 5层 Conv2d: 3→24→36→48→64→64, 每层 BN+ReLU+Dropout
  │         │ 输出: (B*T, 64, H', W')
  └────┬────┘
       │ max-pool(H,W) → (B*T, 64)
       │
       ▼
  ┌─────────┐
  │  LSTM   │ hidden=64, num_layers=1
  │         │ 输出: (B*T, 64)
  └────┬────┘
       │ z (B*T, 64)
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
  ┌─────────┐      ┌──────────┐       ┌──────────┐
  │ Q_MLP   │      │  P_MLP   │       │ StateNet │ (可选)
  │ 64→32→2 │      │ 64→32→2  │       │ 各MLP→1  │
  │ [v̂, δ̂]  │      │ [p₁, p₂] │       │ ds,dd,d, │
  └────┬────┘      └────┬─────┘       │ μ,κ,obs_d│
       │                │             └────┬─────┘
       ▼                │                  │
  ┌──────────┐          │                  │
  │ 导数层    │          │                  │
  │ q =      │          │                  │
  │ -(v̂-v_gt)│          │                  │
  │   /0.1   │          │                  │
  └────┬─────┘          │                  │
       │                │                  │
       │    ┌───────────┘                  │
       │    │                              │
       ▼    ▼                              ▼
  ┌──────────────────────────────────────────┐
  │          BarrierNet QP 层                 │
  │                                          │
  │  输入:                                   │
  │    Q = I (2×2)                           │
  │    q = 参考控制  (来自导数层)              │
  │    G,h = dCBF约束 (来自 p + 状态)         │
  │        - 避障 (用 p₁,p₂)                 │
  │        - 车道保持左 (固定 α₁=2,α₂=1)      │
  │        - 车道保持右 (固定 α₁=2,α₂=1)      │
  │                                          │
  │  求解: min ½u^TQu + q^Tu  s.t. Gu≤h     │
  │  输出: [u_jerk, u_steer]                 │
  └────────────────┬─────────────────────────┘
                   │
                   ▼
  final_out = concat([v̂, δ̂, u_jerk, u_steer] + state_estimates)
                   │
                   ▼
              损失计算
```

---

## 二、损失函数的完整传播路径

### 2.1 `_compute_loss()` 的精确逻辑

```python
def _compute_loss(self, pred, batch):
    state = batch[1]   # (B, T, 7): [s, d, μ, v, a, δ, κ]
    obs   = batch[2]   # (B, T, 2): [Δs, d_obs]
    label = batch[3]   # (B, T, 4): [v_gt, δ_gt, a_gt, ω_gt]

    # output_mode = ['v', 'delta', 'a', 'omega']  ← model_type='deri'
    # pred 按此顺序排列

    losses = {}
    loss_all = 0.0

    # --- 运动状态损失 (Level 2 监督) ---
    pred_v     = pred[:,:,0]     # v̂ 来自 Q_MLP
    label_v    = label[:,:,0]    # v_gt 来自 NMPC
    loss_v     = MSE(pred_v, label_v)
    loss_all  += loss_coef_v * loss_v

    pred_delta = pred[:,:,1]     # δ̂ 来自 Q_MLP
    label_delta = label[:,:,1]   # δ_gt 来自 NMPC
    loss_delta = MSE(pred_delta, label_delta)
    loss_all  += loss_coef_delta * loss_delta

    # --- 控制损失 (Level 4 监督) ---
    pred_a     = pred[:,:,2]     # û_jerk 来自 BarrierNet QP
    label_a    = label[:,:,2]    # a_gt 来自 NMPC
    loss_a     = MSE(pred_a, label_a)
    loss_all  += loss_coef_a * loss_a

    pred_omega = pred[:,:,3]     # û_steer 来自 BarrierNet QP
    label_omega = label[:,:,3]   # ω_gt 来自 NMPC
    loss_omega = MSE(pred_omega, label_omega)
    loss_all  += loss_coef_omega * loss_omega

    # --- 状态估计损失 (Level 1 监督) --- 仅当 StateNet 启用
    # ds: 带边界掩码（在 [ds_bound[0], ds_bound[1]] 范围内才计入损失）
    # dd, d, mu, kappa, obs_d: 标准 MSE
    # obs_d: 当 ds 超出范围时掩码掉（障碍物太远时不监督）

    losses['loss/all'] = loss_all
    return losses
```

### 2.2 梯度传播路径

```
                    L_total
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
    L_control      L_motion           L_state
  (a, ω 的 MSE)  (v, δ 的 MSE)    (ds,dd,d,μ 的 MSE)
        │              │                  │
        ▼              │                  │
  ∂L/∂u_jerk ─────────┤                  │
  ∂L/∂u_steer         │                  │
        │              │                  │
        ▼              │                  │
  ┌─────────────┐      │                  │
  │ OptNet QP   │      │                  │
  │ ∂u*/∂Q = 0  │      │                  │
  │ ∂u*/∂q ──────────→ Q_MLP ←── L_motion┤
  │ ∂u*/∂G ──────────→ P_MLP             │
  │ ∂u*/∂h ──────────→ P_MLP             │
  │             通过 KKT 条件反向传播       │
  └─────────────┘                         │
        │                                 │
        ▼                                 ▼
  P_MLP ← L_control 通过 ∂u*/∂h       Q_MLP ← L_motion + L_control 通过 ∂u*/∂q
        │                                 │
        └────────────┬────────────────────┘
                     ▼
                  LSTM
                     │
                     ▼
                    CNN
```

**关键点**：
- **Q_MLP 收到双重梯度**：直接的运动损失 $\partial\mathcal{L}_{\text{motion}}/\partial q$ + 通过 QP 的控制损失 $\partial\mathcal{L}_{\text{control}}/\partial u^* \cdot \partial u^*/\partial q$
- **P_MLP 只通过 QP 收到梯度**：$\partial\mathcal{L}_{\text{control}}/\partial u^* \cdot \partial u^*/\partial h$（没有直接监督 p₁,p₂）
- **CNN+LSTM 收到全部梯度**：三条路径汇聚
- **StateNet MLP 只收到状态损失**：梯度不通过 QP

---

## 三、dCBF 约束的完整数学构造

### 3.1 避障 HOCBF（相对度 m=2）

```
原始约束: b_obs = Δs² + (d - d_obs)² - r_D² ≥ 0       (r_D = 7.9m)

HOCBF 序列 (m=2):
  ψ₀ = b_obs
  ψ₁ = ψ̇₀ + p₁·α₁(ψ₀)                                   ← p₁ 来自 P_MLP
  ψ₂ = ψ̇₁ + p₂·α₂(ψ₁)                                   ← p₂ 来自 P_MLP

dCBF 约束 (代入动力学计算的 Lie 导数):
  L_f² b + [L_g L_f b]·u + (p₁+p₂)·ḃ + p₁·p₂·b ≥ 0

代码对应:
  h_obs = Lf2b + (p₁+p₂)*barrier_dot + p₁*p₂*barrier
         └─┬──┘   └──────┬──────┘   └───┬───┘
          L_f²b     (p₁+p₂)·ψ̇₀       p₁·p₂·ψ₀

  G_obs = -[L_g L_f b]                                   ← 取负，因为 QP 用 Gu≤h 形式
```

其中 $\alpha_1(\psi_0) = \psi_0$, $\alpha_2(\psi_1) = \psi_1$（线性 K 类函数），因此：
- $p_1(z) \cdot \alpha_1(\psi_0) = p_1 \cdot b$
- $p_2(z) \cdot \alpha_2(\psi_1) = p_2 \cdot (\dot{b} + p_1 b)$
- 展开：$L_f^2 b + L_g L_f b \cdot u + (p_1+p_2)\dot{b} + p_1 p_2 b \geq 0$

### 3.2 车道保持 HOCBF（固定参数，非可学习）

```
左边界: b_left = 2.0 - d ≥ 0
右边界: b_right = d + 2.0 ≥ 0

使用固定 α₁=2, α₂=1:
  h_left  = Lf2b_left  + 2*barrier_dot_left  + 1*barrier_left
  h_right = Lf2b_right + 2*barrier_dot_right + 1*barrier_right

注意: 车道保持不使用 p_i(z)，保持了确定性的安全保证
```

### 3.3 Lie 导数的完整表达式（避障）

```
车辆参数: lrf = 0.5, lr = 2.78

运动学中间量:
  β = arctan(lrf · tan(δ))
  cos_mu_beta = cos(μ + β)
  sin_mu_beta = sin(μ + β)
  μ̇ = v/lr · sin(β) - κ · v · cos_mu_beta / (1 - d·κ)

b_obs = Δs² + (d - d_obs)² - r_D²

一阶 Lie 导数:
  ḃ_obs = 2·Δs·v·cos_mu_beta/(1-d·κ) + 2·(d-d_obs)·v·sin_mu_beta

二阶 Lie 导数 L_f²b:
  包含 cos_mu_beta², sin_mu_beta², μ̇, κ 的耦合项

L_g L_f b (控制系数矩阵):
  第 1 列 (对 u_jerk): 2·Δs·cos_mu_beta/(1-d·κ) + 2·(d-d_obs)·sin_mu_beta
  第 2 列 (对 u_steer): 含 lrf/cos²δ/(1+(lrf·tanδ)²) 因子
```

---

## 四、三种模型类型 (model_type)

| model_type | 输出 | q (QP代价向量) 含义 | 导数层 |
|------------|------|--------------------|--------|
| `'deri'` | `[v̂, δ̂, u_jerk, u_steer]` | $q = -(v̂-v_{gt})/0.1$ | 有（v̂,δ̂ → 数值微分 → 参考控制误差） |
| `'inte'` | `[a, ω]` (直接控制) | 无导数层，q 直接用 | 无 |
| `'direct'` | `[a, ω]` (直接控制) | q 直接用 | 无 |

**论文使用 `'deri'` 模式**。核心设计是 Q_MLP 预测速度 v̂ 和转向角 δ̂（有直接的 GT 监督），然后导数层隐式推导参考加速度和参考转向率（通过数值微分），QP 层在此基础上做安全修正。这与车辆的物理层级（位置→速度→加速度→加加速度）完全对应。

---

## 五、Q_MLP 和 P_MLP 的输入来源与监督信号

| 分支 | 输入 | 输出 | 直接监督 | 通过 QP 的间接监督 |
|------|------|------|---------|-------------------|
| **Q_MLP** | LSTM 特征 z (B*T, 64) | [v̂, δ̂] (B*T, 2) | MSE(v̂, v_gt) + MSE(δ̂, δ_gt) | $\partial\mathcal{L}_{\text{ctrl}}/\partial u^* \cdot \partial u^*/\partial q$ |
| **P_MLP** | LSTM 特征 z (B*T, 64) | [p₁, p₂] (B*T, 2), ×4, Sigmoid | **无直接监督！** | $\partial\mathcal{L}_{\text{ctrl}}/\partial u^* \cdot \partial u^*/\partial h \cdot \partial h/\partial p$ |
| **StateNet MLPs** | CNN 特征 或 LSTM 特征 | [ds, dd, d, μ, κ, obs_d] | MSE with GT | 无（梯度在 detach+clamp 处截断） |

**P_MLP 的关键设计**：
- **没有直接监督**：p₁,p₂ 的值完全由"QP 输出的控制是否匹配 GT 控制"这个间接信号驱动
- **×4 因子**：扩大 Sigmoid 后的输出范围（Sigmoid 输出 [0,1]，乘 4 后 [0,4]）
- **Sigmoid**：保证 p_i > 0（CBF 理论要求），同时提供 Lipschitz 连续性
- 如果 QP 被过度约束（控制输出偏离参考太多）→ 梯度推动 P_MLP 增大 p_i 值（收紧约束）→ 让 QP 输出更接近参考

---

## 六、StateNet 的稳定性设计

```python
# 1. 非 detach 版本 — 用于损失计算
d_non_detach = d

# 2. detach — 切断向 CNN/LSTM 的梯度
d_detached = d.detach()

# 3. clamp — 限制在 GT ±5% 范围内
tol = 0.05
d_clamped = clamp(d_detached, gt_d*(1-tol), gt_d*(1+tol))

# 4. 用 clamp 后的值构造 CBF 约束（不会反向传播到 CNN/LSTM）
#    但用非 detach 的值计算状态损失 L_state
```

**为什么这样设计？**
- StateNet 预测不准确时（尤其是训练早期），不准确的 d, μ 会导致 CBF 约束构造错误 → QP 无解或输出异常
- 用 `detach + clamp` 保证 CBF 约束**总在真值附近**（不传播状态估计误差到 QP）
- 状态估计的学习通过独立的 MSE 损失进行，不影响安全层

---

## 七、训练 Pipeline

### 7.1 数据格式

```
每个 batch:
  batch[0]: img_seq    (B, T, 3, H, W)   图像序列
  batch[1]: state_seq  (B, T, 7)          [s, d, μ, v, a, δ, κ]
  batch[2]: obs_seq    (B, T, 2)          [Δs, d_obs]
  batch[3]: ctrl_seq   (B, T, 4)          [v_gt, δ_gt, a_gt, ω_gt]
```

### 7.2 训练 step

```python
def training_step(self, batch, batch_idx):
    rnn_state = self.get_initial_state(batch_size)  # LSTM 初始状态 (0向量)
    pred, rnn_state = self(batch, rnn_state, solver='QP')
    losses = self._compute_loss(pred, batch)

    for k, v in losses.items():
        self.log(f'train/{k}', v)

    return losses['loss/all']  # → Adam 优化器
```

### 7.3 优化器

```python
optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
```

所有参数（CNN, LSTM, Q_MLP, P_MLP, StateNet MLPs）联合优化。

---

## 八、关键数值参数速查表

| 参数 | 值 | 来源 |
|------|-----|------|
| CNN 层数 | 5 | `cnn_params` |
| CNN 第一层输入通道 | 3 (RGB) | `[3, 24, 5, 2, 2]` |
| CNN 最终输出通道 | 64 | `[64, 64, 3, 1, 1]` |
| CNN dropout | 0.3 | `cnn_dropout` |
| LSTM hidden size | 64 | `lstm_size` |
| LSTM 层数 | 1 | `nn.LSTM` 默认 |
| Q_MLP 结构 | 64→32→32→2 | `[32, 32, 2]` |
| Q_MLP dropout | 0.3 | `q_mlp_dropout` |
| P_MLP 结构 | 64→32→32→2 | `[32, 32, 2]` |
| P_MLP 输出激活 | Sigmoid | `with_output_norm=True` |
| P_MLP 缩放因子 | ×4 | `p = self._p_mlp(z)*4` |
| StateNet MLPs | 64→32→32→1 (各) | `[32, 32, 1]` |
| StateNet 激活 | ELU | `elu` |
| StateNet clamp 容差 | ±5% | `state_tol=0.05` |
| 导数层缩放因子 | 1/0.1 = 10 | `(v_delta - state)/0.1` |
| 学习率 | 1e-3 | `lr` |
| 轴距 lr | 2.78 m | `lr = 2.78` |
| 重心比例 lrf | 0.5 | `lrf = 0.5` |
| 圆盘半径 | 7.9 m | `7.9**2` |
| 车道保持阈值 | 2.0 m | `lf_cbf_threshold` |
| 车道 CBF α₁ | 2.0 | `2*barrier_dot` |
| 车道 CBF α₂ | 1.0 | `1*barrier` |
| 损失函数 | MSE | `nn.MSELoss()` |
| 设备 | 2080Ti | 硬件 |

---

## 九、关键设计决策及其代码体现

| 设计决策 | 代码位置 | 原因 |
|---------|---------|------|
| Q_MLP 和 P_MLP 共享 LSTM 特征 | `z = self._lstm(...); q = self._q_mlp(z); p = self._p_mlp(z)` | 减少参数，让 p 也能利用运动信息 |
| P_MLP ×4 | `p = self._p_mlp(z)*4` | Sigmoid 输出 [0,1] 太小，缩放增强灵活性 |
| 导数层的 q 取负 | `q = -((v_delta - state_seq[:,3:5])/0.1)` | QP 最小化时，q<0 → 推 u>0（加速），方向正确 |
| StateNet detach+clamp | `d = torch.clamp(d.detach(), ...)` | 不准确的估计不要破坏 CBF 约束 |
| 训练时无控制边界 | `QPFunction(...)` 无 u_min, u_max | 避免早期不可行 |
| 车道 CBF 固定 α | `2*barrier_dot + 1*barrier` | 车道保持有明确物理边界，不需要可学习参数 |
| 避障 CBF 可学习 p | `(p₁+p₂)*barrier_dot + p₁*p₂*barrier` | 避障需要随环境自适应松紧 |
| LSTM batch_first=True | `nn.LSTM(..., batch_first=True)` | 输入 (B,T,64) 而非 (T,B,64) |
| 无 batchnorm 在 LSTM 后 | Q_MLP 和 P_MLP 无 BN | LSTM 输出已归一化，加 BN 破坏时序信息 |
| ds 损失掩码 | `mask = ~((pred>bound)&(label>bound) \| ...)` | 状态值在合理范围外时不惩罚 |
