# CacheCodec

新版实验方案见 `CacheCodec_new_project_experiment_plan.md`。新版主线把
CacheCodec 定义为 Sharer→Receiver 的通信 codec：JCB（joint K/V bottleneck）
后接序列维 DCT、固定非线性量化和 zlib6；不使用 adaptive quant table、learned
group alpha 或 rate loss。原有 LCF/adaptive recipe 保留为历史兼容路径，新实验请
使用 `recipe/train_recipe/cachecodec_concat_jcb_external.json` 和对应 eval recipe；
六组需要训练的配置可由 `bash/train/run_cachecodec_v2_matrix_8gpu.sh` 串行执行。

QAT 训练组使用 `cachecodec_concat_lcf_qat_external.json` 和
`cachecodec_fusion_lcf_qat_external.json`：训练 forward 只执行 DCT + fixed fake
quant + STE，不执行 zlib6；真实 int16/zlib6 只在对应 `*_qat_quant_external.yaml`
评测中执行。

所有大文件通过环境变量放在仓库外：

```bash
export CACHECODEC_MODEL_ROOT=/path/to/models
export CACHECODEC_DATA_ROOT=/path/to/data
export CACHECODEC_RUN_ROOT=/path/to/cachecodec_runs
```

配置中的 `${CACHECODEC_*}` 会在训练和评测入口自动展开；checkpoint、数据和结果
不会写入 Git 工作树。

## 旧版方案说明

> **文档用途。** 本文档是当前讨论版本的技术规范，用于核对 pipeline、张量形状、训练与部署边界；它不是论文正文，也不代表所有模块均已实现或完成实验验证。
>
> **历史兼容说明。** 下方旧版章节记录 `LCFProjectedKV + adaptive packet` 实现，仅供加载历史 checkpoint；新版主线、训练/评测入口和参数以 `CacheCodec_new_project_experiment_plan.md` 与 `docs/cachecodec_v2_migration.md` 为准。

> 当前代码结构、训练阶段和四个 benchmark 的逐步核对清单见 [`docs/当前代码结构与实验流程_中文.md`](docs/当前代码结构与实验流程_中文.md)。

## 1. 要解决的问题

设一个冻结的 **sharer** LLM 与一个冻结的、结构可能不同的 **receiver** LLM 共同完成任务。两者可以承担不同的 model role，并接收不同的 role-specific 输入：sharer 处理源上下文 `x_s`，receiver 处理自身输入 `x_r`。文本到文本（T2T）通信需要将 sharer 的内部状态自回归地解码为文本，再由 receiver 重新编码；cache-level communication 则希望直接把与源上下文相关的 KV 表示交给 receiver 使用。

本方案不要求 sharer 与 receiver 使用相同的 token 序列，也不执行 token-to-token 的位置对应或逐 token 表示相加。Sharer 的 cache 被重建为 receiver 可用的 virtual prefix，随后 receiver 在一次 prefill 中将该 prefix 与自身的 `x_r` 一同处理。因此，跨模型之间需要固定的是参与通信的 layer route 和 receiver-side prefix 协议，而不是 tokenizer alignment；`S_s` 与 `S_r` 分别表示两侧序列长度。

直接传输 sharer 的完整 KV cache 并不可取：其大小随序列长度、层数、KV heads 和 head dimension 增长。在实际带宽有限时，通信时间由 payload 决定的一部分可能抵消 cache-level communication 避免文本生成所带来的收益。

本方案研究的具体问题是：

> 如何把 sharer 的 KV cache 编码为一个可真实序列化、可在 receiver 端重建的紧凑消息，并通过 receiver 的下游任务损失学习“哪些失真可以接受、哪些 group 值得保留”？

这里的目标是减少通信 payload，并在未来通过真实系统测量考察端到端时延；在尚未测量编码/解码和网络时间前，**不应直接声称已降低端到端时延**。

## 2. 设计原则与边界

### 2.1 发送端优先的通信架构

压缩、量化和码率选择尽可能位于 sharer 侧：

- 发送的是熵编码后的紧凑 packet，而不是高维 receiver-side KV；
- receiver 只执行确定性的 packet 解码、反量化、IDCT 与两条轻量 K/V decoder，并将重建 cache 拼接为 prefix cache；
- 所有可学习模块在离线后训练阶段优化；部署时不需要 receiver 运行 allocator 或搜索量化级别。

这并不意味着 receiver 端完全没有计算，而是避免把选择、投影、融合等重型决策留给每次接收消息时的 receiver。

### 2.2 本方案的三层压缩逻辑

1. **Channel bottleneck：** 先对每层 K/V 的 `H×D` 维进行可学习压缩，去除跨 head / channel 的冗余。
2. **Sequence-domain coding：** 再只沿 sequence 维执行 DCT，将时序变化表示为频域系数；DCT 本身不改变元素个数，而是提供更利于非均匀量化和熵编码的表示。
3. **Group-wise rate allocation：** 对每个 `layer × {K,V}` group 选择离散量化步长，让不同 group 的码率随其频谱与训练中体现的下游敏感性而变化。

这三个阶段解决的是不同维度的冗余，不能混为“DCT 直接压缩了 KV”。真正减少字节数的是 latent bottleneck、量化和熵编码。

### 2.3 不在当前方案中假设的结论

- 不假设所有高频系数都是噪声；只检验低频/高频在当前任务和模型对上的可压缩性差异。
- 不将 allocator 输出解释为因果的“信息重要性”；它首先是优化得到的码率决策。
- 不把 `\hat R` 当作真实 packet bits，也不把训练 surrogate 的降低直接等同于真实 payload 的降低。
- 不声称比 T2T 的 payload 更小；两者传输单位和生成过程不同，应以完整系统时间比较。

## 3. 符号、示例配置与分组

### 3.1 基本符号

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `S_s` | sharer source context 对应的 cache/prefix sequence length |
| `S_r` | receiver 自身输入序列的长度 |
| `S_total=S_s+S_r` | receiver prefill 后的总 cache sequence length |
| `x_s`, `x_r` | sharer 的源上下文与 receiver 的 role-specific 输入 |
| `y` | OpenHermes assistant response tokens，用于 teacher-forcing CE |
| `L_s`, `L_r` | sharer、receiver 的参与通信层数 |
| `H_s,D_s` | sharer 的 KV head 数与每 head 维度 |
| `H_r,D_r` | receiver 的 KV head 数与每 head 维度 |
| `C_s=H_sD_s` | sharer 单个 K 或 V 的展平 channel 宽度 |
| `C_r=H_rD_r` | receiver 单个 K 或 V 的展平 channel 宽度 |
| `d` | shared latent 宽度 |
| `d_k=d_v=d/2` | K / V latent 宽度（本版本默认各为 64） |
| `g=(\ell,t)` | 通信 group，`\ell` 是逻辑接收层，`t∈{K,V}` |
| `\mathcal A` | 可选离散量化 multiplier 集合 |

### 3.2 贯穿全文的示例维度

为避免“压缩维度”含义含混，下文使用一个示例，而不是绑定任何最终模型配置：

```text
S_s = 2048
S_r = receiver input length（由具体样本决定）
S_total = S_s + S_r
H_s D_s = C_s = 1024
H_r D_r = C_r = 128
d = 128, d_k = d_v = 64

一层 sharer K: [B, H_s, S_s, D_s] -> [B, S_s, 1024]
一层 sharer V: [B, H_s, S_s, D_s] -> [B, S_s, 1024]
K/V concat:                         [B, S_s, 2048]
shared latent:                       [B, S_s, 128]
K latent / V latent:                [B, S_s, 64] / [B, S_s, 64]
receiver K / V prefix output:       [B, S_s, 128] / [B, S_s, 128]
final receiver cache after prefill: [B, H_r, S_total, D_r]
```

`C_s` 与 `C_r` 可以不同，因此这不是同构 cache 的直接截断；每个逻辑 receiver layer 的 LCF encoder/decoder 同时承担跨模型几何转换与通信 bottleneck 的角色。

### 3.3 层路由与 group 定义

异构模型的层数可能不同，因此在 LCF 前使用一个**固定的 terminal-alignment 层路由** `r(\ell)`。从两侧网络的末层开始向前逐层匹配：一个 receiver layer 对应一个 source layer，不做多 source layer 聚合，也不学习路由。若 `L_s\ge L_r`，当前实现采用：

\[
r(L_r-1-j)=L_s-1-j,\qquad j=0,\ldots,L_r-1,
\]

即 receiver 的最后一层首先对应 source 的最后一层，然后依次向前对应。若 `L_s<L_r`，source 的末端 `L_s` 层仍对应 receiver 的末端 `L_s` 层，receiver 前部未覆盖的层统一复用 source 第 0 层。该回落是显式协议，不是隐式 clamp；因此 route 始终覆盖全部 receiver 层。

量化 group 采用：

\[
g=(\ell,t),\qquad t\in\{K,V\}.
\]

因此最终每个 group 都有一个独立离散决策 `\alpha_{\ell,K}` 或 `\alpha_{\ell,V}`。**当前版本不再声称 head-wise alpha。** `H×D` 的差异首先由可学习 LCF bottleneck 处理；是否再做 head-wise 分配应作为后续扩展/消融，而非当前主方法。

## 4. 总体数据流

```text
                             OFFLINE TRAINING

input x_s -> frozen sharer -> per-layer K/V -> LCF encoder -> K/V projections
                                                     |                    |
                                                     |                 1D DCT (S)
                                                     |                 |
frozen receiver <- prefix cache <- K/V decoders <- IDCT <- dequant <- STE quantizer
       |                                                             ^
       +---------------------- task CE loss ------------------------+
                                      + lambda * R-hat

                               DEPLOYMENT

sharer KV -> LCF encoder -> K/V projections -> DCT -> scale -> alpha argmax
          -> quantize -> metadata + entropy-coded symbols -> packet
          -> receiver: decode -> dequantize -> IDCT -> K/V decoders -> prefix cache
          -> frozen receiver generation/scoring
```

训练与部署的共同点是：receiver 消费的都是**量化后重建**的 cache。训练使用可导的近似量化和 `\hat R`；当前 adaptive concat eval 已使用真正的离散符号、序列化 packet 和 zlib lossless round-trip。固定 `CacheJPEG` 仍保留为独立 fixed-quant 对照，不得标记为 adaptive 主结果。

## 5. 模块一：K/V 联合的 LCF-first latent bottleneck

### 5.1 输入预处理

对某个被路由到逻辑层 `\ell` 的 sharer cache：

1. 对 Key 先做 de-RoPE，使其回到位置无关的表示空间；Value 不需要这一操作。
2. 将 `[B,H_s,S_s,D_s]` 转置/展平为 `[B,S_s,C_s]`。
3. 在最后一个 channel 维拼接 K 与 V：

\[
X_\ell=\operatorname{Concat}(K^{\text{de-RoPE}}_\ell,V_\ell)
\in\mathbb{R}^{B\times S_s\times 2C_s}.
\]

K/V 在 bottleneck 前共享编码器，使 encoder 可以利用二者的相关性；这不等价于假设 K 和 V 完全相同。

### 5.2 Per-layer joint encoder 与 K/V-specific projections

对每个逻辑 receiver layer `\ell` 使用一套 LCF encoder；**shared** 仅表示该层内 K/V 共用同一个 joint encoder，不表示所有层共用参数。示例配置使用：

\[
E_{\phi,\ell}:2C_s\rightarrow d\rightarrow4d\rightarrow d.
\]

例如 `2048 → 128 → 512 → 128`，中间使用非线性（如 GELU/SiLU；最终实现需固定一种）。令

\[
z_\ell=E_{\phi,\ell}(X_\ell)\in\mathbb{R}^{B\times S_s\times d}.
\]

随后，使用两个独立的可学习投影将完整的 joint latent 映射为 K/V 专用 latent：

\[
z_{\ell,K}=P_{\psi,K,\ell}(z_\ell),\qquad
z_{\ell,V}=P_{\psi,V,\ell}(z_\ell),
\]
\[
P_{\psi,K,\ell},P_{\psi,V,\ell}:d\rightarrow d/2.
\]

默认 `d=128`，因此两个 latent 均为 `[B,S_s,64]`。与固定 channel split 不同，两个投影均从完整的 128 维 joint latent 中学习提取各自所需的信息，而非直接截取前后 64 个 channel。投影参数与 joint encoder 一同在端到端任务损失约束下优化，为后续 K/V 独立的频域编码和 receiver-side 重建提供专用表示。

### 5.3 K/V-specific decoders 和 receiver-compatible KV

从通信端恢复的 K/V latent 不再 concat。每个逻辑 receiver layer 使用两条独立 decoder，分别将对应 latent 映射到 receiver 单个 K 或 V 的展平宽度：

\[
\tilde K_\ell=D_{\omega,K,\ell}(\hat z_{\ell,K}),\qquad
\tilde V_\ell=D_{\omega,V,\ell}(\hat z_{\ell,V}),
\]

其中：

\[
D_{\omega,K,\ell},D_{\omega,V,\ell}:d/2\rightarrow2d\rightarrow C_r.
\]

按示例 `d=128,C_r=128`，每条分支为：

```text
 K latent [B,S_s,64] -> Decoder_K: 64 -> 256 -> 128 (= C_r) -> receiver K prefix
 V latent [B,S_s,64] -> Decoder_V: 64 -> 256 -> 128 (= C_r) -> receiver V prefix
```

`D_{\omega,K,\ell}` 与 `D_{\omega,V,\ell}` 具有相同结构但**不共享参数**，且不同逻辑层默认也不共享参数；不存在它们之后的 K/V output heads。K/V 在每层 encoder 前联合建模，projection 后从量化到 receiver 映射均保持独立路径，从而与每个 `layer×K/V` group 的独立 alpha 相一致。

最后将 `[B,S_s,C_r]` reshape 为 `[B,H_r,S_s,D_r]`。Key 的位置处理必须遵循以下固定协议：

1. sharer 端 Key 先去除 sharer RoPE，Value 不做 RoPE 操作；
2. decoder 输出的 receiver Key 在 receiver 端按照通信 prefix 在 receiver 序列中的实际位置重新施加 receiver RoPE；
3. `position_ids` 必须与该 prefix 的布局一致。若通信 cache 放在 receiver 自有上下文之前，则通信 token 使用 `0,...,S_s-1`，receiver 后续 token 的 position 从 `S_s`（或已有 receiver context 长度加 `S_s`）开始；
4. attention mask、KV cache 拼接顺序和后续 token 的 position offset 必须使用同一约定；Value 始终不施加 RoPE。

重建 K/V 与 receiver 自有 cache 按该 prefix 协议 concat，作为 `past_key_values` 供 receiver 使用。

### 5.4 为什么 LCF 放在 DCT 之前

本方案选择 **LCF-first**：先把 `H×D` 压到固定 latent，再沿 `S_s` 做 DCT。

- LCF 先减少每个 token 的 channel 数，因此待传输频域系数的数量已经下降；
- DCT 只作用于 sequence 轴，对每个 latent channel 独立线性变换，不会破坏 `H×D` bottleneck 的定义；
- K/V 在 LCF 中联合建模，但随后各自量化，允许它们采取不同传输精度。

是否优于“DCT-first”是实验问题，建议将 DCT-first 或 no-LCF 作为消融，而不预设结论。

## 6. 模块二：逐 K/V 的序列频域量化

### 6.1 1D DCT 与尺度归一化

对每个 group 的 latent `z_g∈R^{B×S_s×64}`，仅在长度为 `S_s` 的序列轴施加正交 1D-DCT：

\[
c_g=\operatorname{DCT}_{S_s}(z_g).
\]

然后以 group 为单位计算尺度 `s_g`（例如 RMS；需明确是在 batch、sequence、channel 的哪些轴上统计），并归一化：

\[
u_g=c_g/(s_g+\epsilon).
\]

训练时该尺度的梯度路径应在实现中明确：若使用 `\operatorname{stopgrad}(s_g)`，其作用是稳定量化尺度，而不是让模型通过任意缩放规避量化代价。该选择需要作为实现超参数记录，并做 `no-detach-scale` 消融（若资源允许）。

### 6.2 离散量化强度

对 group `g`，allocator 从候选集合中选择：

\[
\mathcal A=\{0.125,0.25,0.5,1,2\}\quad\text{（当前候选；非最终定论）}.
\]

为消除“alpha 大小”在不同代码中含义相反的风险，本论文必须固定以下约定：

\[
\Delta_g=\alpha_g\Delta_0,\qquad
q_g=\operatorname{round}(u_g/\Delta_g),\qquad
\hat u_g=q_g\Delta_g.
\]

在该约定下，`α` 越大，步长越大、量化越粗，通常码率更低且失真可能更高。若实际代码采用了倒数形式，必须相应改写符号与文字，不能只保留“alpha 越大/越小”的口头表述。

训练使用 straight-through estimator (STE)：前向近似离散 `round`，反向把其局部梯度近似为恒等映射，使任务 CE 可以更新 encoder、decoder 和 allocator 的参数。部署时使用真正的 `round` 和整数符号。

### 6.3 重建

反量化、恢复尺度和 IDCT 为：

\[
\hat c_g=s_g\hat u_g,\qquad
\hat z_g=\operatorname{IDCT}_{S_s}(\hat c_g).
\]

得到的 `\hat z_{\ell,K}`、`\hat z_{\ell,V}` 分别进入第 5 节的 K/V decoder。DCT/IDCT 在无量化时近似可逆；下游性能损失主要来自 latent bottleneck 与量化，而非 DCT 变换本身。

## 7. 模块三：离散码率 allocator

### 7.1 目标

allocator 的职责不是显式预测“哪个 group 最重要”，而是输出每个 group 对应 alpha 的 categorical logits：

\[
\mathbf a_g\in\mathbb R^{|\mathcal A|}.
\]

它的输入应同时包含：

- 当前 group 的频谱统计，反映该 group 的系数分布；
- layer 与 K/V 身份，避免模型把不同 group 视为可交换对象；
- 所有 group 的全局 context，反映有限总通信预算下的相互关系。

### 7.2 频谱统计与 local encoder

把 `u_g` 的频率轴划分为 8 个频带。每个频带提取 4 个统计量（最终需与代码一致，例如均值绝对值、RMS/能量、零/近零比例、熵或峰度），得到：

```text
8 frequency bands × 4 statistics = 32-dimensional spectral descriptor
```

在当前推荐结构中，组身份在 local encoder 前注入：

```text
spectral descriptor: 32
layer embedding:     16
K/V embedding:        4
concat:              52
LocalMLP:            52 -> 64 -> 128
```

这样 local feature 已经知道“这是哪一层的 K 或 V”，比将身份信息放在 global stage 后再注入更一致。

> 注：先前讨论过 `32→64→128` 再 concat identity 的版本。该版本也能工作，但 identity 不能参与最初的谱特征解释；当前以 `52→64→128` 为修正后的规范。

### 7.3 跨 group context 与 alpha logits

将所有 `G=2L_r` 个 group 的 local features 记为 `h_g∈R^{128}`。第一版使用 mean pooling 建立简洁的全局上下文：

\[
c=\operatorname{GlobalMLP}\left(\frac{1}{G}\sum_{g=1}^{G}h_g\right)\in\mathbb R^{128}.
\]

随后对每个 group 预测五个 alpha logits：

```text
[h_g ; c]                    : 128 + 128 = 256
AlphaMLP                     : 256 -> 128 -> |A| (= 5)
output                       : logits a_g
```

mean pooling 的优点是维度固定、开销小；缺点是无法表达强的 group-to-group pairwise interaction。cross-group attention 是合理的后续扩展或消融，但不应在第一版主方法中无证据加入复杂度。

### 7.4 Gumbel-Softmax：训练与部署的一致性

令 `π_g=softmax(a_g)`。训练时采样 Gumbel 噪声并形成温度为 `τ` 的 relaxed categorical：

\[
y_{g,i}=\frac{\exp((a_{g,i}+\gamma_i)/\tau)}
{\sum_j\exp((a_{g,j}+\gamma_j)/\tau)},\quad
\gamma_i=-\log(-\log U_i),\ U_i\sim\operatorname{Uniform}(0,1).
\]

使用 `hard=True` 时：

- 前向用 one-hot 的 `\operatorname{onehot}(\arg\max_i y_{g,i})` 选择一个实际离散 alpha；
- 反向通过 soft `y_g` 的梯度近似更新 logits；
- 因此训练时 receiver 已经看到离散量化的前向效果，而 allocator 仍能学习。

部署不采样 Gumbel 噪声，直接取：

\[
\alpha_g=\mathcal A[\arg\max_i a_{g,i}].
\]

候选集中新增一个看似更粗的 alpha 后，模型最终可能更常选择较细的 alpha；这是离散优化景观、softmax 归一化、随机初始化与 `CE+λ\hat R` 权衡共同变化的结果，并不构成 alpha 单调性的反例。必须报告多 seed 的 alpha 分布、真实 bytes 和性能，不能仅由单个 checkpoint 的选择比例解释机制。

## 8. 训练目标、可导码率与真实 payload

### 8.1 训练目标

冻结 sharer 和 receiver，只训练通信 codec 参数：

\[
\Theta=\{E_{\phi,\ell},D_{\omega,K,\ell},D_{\omega,V,\ell},\text{allocator}\}_{\ell=1}^{L_r}.
\]

基本目标：

\[
\mathcal L(\Theta)=\mathcal L_{\mathrm{task}}(y, f_r(x_r;\widehat{\mathrm{KV}}_\Theta(x_s)))
+\lambda\widehat R(\Theta).
\]

当前训练统一使用 OpenHermes 中的 assistant response tokens 作为 teacher-forcing 目标，而不是使用 MCQ 的 gold option label。给定 Sharer 的源上下文 `x_s`、Receiver 的 role-specific 输入 `x_r` 和目标 response
`y=(y_1,\ldots,y_T)`，任务损失定义为

\[
\mathcal L_{\mathrm{task}}
=-\sum_{t=1}^{T}
\log p_{\mathcal M_R}
\left(y_t\mid y_{<t},x_r,\widehat{\mathrm{KV}}_\Theta(x_s)\right).
\]

因此，Sharer 与 Receiver 可以看到不同的输入文本、使用不同的 tokenizer，并承担不同的 model role；训练监督只约束 Receiver 在重建 prefix 条件下生成 OpenHermes response。下游选择题或其他 benchmark 的评测协议与该 response-level 训练目标分开记录，不应改写为 gold-label training。

### 8.2 `z`、`-log2 z` 和 `\hat R`

真实熵编码后的字节数包含整数化、包头、coder 状态和具体符号序列，因此不可直接作为神经网络反向传播的目标。训练中需要 rate surrogate。

若量化符号（或其软近似）为 `q`，概率模型/统计模型给其概率为：

\[
z=p_\psi(q),
\]

则信息论意义上的理想码长是：

\[
-\log_2 z=-\log_2p_\psi(q)\quad\text{bits}.
\]

对所有 group、位置和 latent channel 求和/平均可得：

\[
\widehat R=\sum_{g,n,c}-\log_2p_\psi(\widetilde q_{g,n,c}).
\]

其中 `\widetilde q` 表示允许梯度近似流动的软符号或 STE 符号。`z` 不是原始 KV 值，也不是 zlib 的压缩率；它是“当前符号在训练中采用的概率模型下有多常见”的概率。概率越高，预测码长越短。

实现必须记录 `p_ψ` 的精确定义：是可学习 entropy model、固定分布、由零率/幅值估计的近似，还是其他统计 surrogate。若当前 `\hat R` 未显式建模 `p_ψ`，论文中应只称其为 **differentiable rate proxy**，而不要称为精确 entropy estimate。

梯度路径为：

```text
CE -> reconstructed KV -> IDCT/dequant/STE -> alpha logits, encoder, decoder
R-hat -> soft/STE symbols and alpha probabilities -> alpha logits, encoder
```

真实熵编码器不在梯度图内。

### 8.3 部署 packet 与真实评测率

部署时每个 packet 至少包含（layer route 由 checkpoint manifest 固定校验，不重复写入每个 packet）：

1. 协议与形状信息（版本、batch、`S_s`、latent shape）；
2. 每个 `layer×K/V` group 的 alpha ID；
3. 每个 group 的反量化所需 scale（若 scale 不可由 receiver 复现）；
4. 熵编码后的整数 DCT symbols。

receiver 严格从 packet 解码、反量化、IDCT 和 decoder 重建 KV，不能在评测时绕过 packet 使用训练中的浮点 latent。

令 `B_packet` 为实际序列化 packet 字节数。当前新方案要求的压缩率分母是**同一输入、同一选定 sharer KV 的未压缩原始表示**：

\[
\mathrm{CR}=\frac{B_{\mathrm{raw\ sharer\ KV}}}{B_{\mathrm{packet}}}.
\]

必须固定原始基准的 dtype、纳入的层、K/V 和 sequence length；metadata 是否计入 packet 也必须固定。`\hat R` 只用于训练，`B_packet` 与上述 CR 只用于真实评测。二者应报告相关性/校准误差，但不要求数值相等。

## 9. 训练与推理伪代码

### 9.1 训练（每个 batch）

```python
# f_s and f_r are frozen.
# x_s and x_r may differ because the two models can play different roles.
source_kv = sharer(x_s)
reconstructed_groups = []

for logical_receiver_layer ell:
    source_K, source_V = route(source_kv, ell)
    source_K = de_rope(source_K)
    X = concat(flatten_heads(source_K), flatten_heads(source_V))

    z = encoder[ell](X)
    z_K = K_projection[ell](z)  # learnable 128 -> 64 projection
    z_V = V_projection[ell](z)  # learnable 128 -> 64 projection
    c_K, c_V = dct_over_sequence(z_K), dct_over_sequence(z_V)
    scale_K, scale_V = compute_group_scales(c_K, c_V)
    u_K, u_V = normalize(c_K, scale_K), normalize(c_V, scale_V)
    statistics = spectral_statistics(u_K, u_V)

logits_all = allocator(statistics, layer_ids, kv_ids)

for group g in all layer_times_KV groups:
    alpha_g = hard_gumbel_softmax(logits_all[g], candidates=A)
    q_g = ste_quantize(u_g, alpha_g)
    c_hat_g = dequantize(q_g, scale_g)
    zhat_g = idct_over_sequence(c_hat_g)
    reconstructed_groups.append(zhat_g)

receiver_prefix = assemble_prefix(
    K_decoder[ell](zhat_K_per_layer), V_decoder[ell](zhat_V_per_layer)
)
task_loss = response_cross_entropy(receiver(x_r, prefix=receiver_prefix), y)
rate_loss = differentiable_symbol_rate(q)
loss = task_loss + lambda_rate * rate_loss
loss.backward()
optimizer.step()
```

### 9.2 部署/评测

```python
source_kv = sharer(x_s)
for each group:
    compute encoder latent, DCT coefficients, and group scale
    compute normalized spectral statistics
    alpha_g = A[argmax(allocator_logits_g)]
    q_g = integer_quantize(normalized_coefficients, alpha_g)
    packet.write(alpha_id_g, scale_g, q_g)

bytes_sent = entropy_encode(packet)
received_packet = entropy_decode(bytes_sent)
reconstructed_prefix = deterministic_decode(received_packet)
prediction = receiver(x_r, prefix=reconstructed_prefix)
```

部署路径必须做真实 round trip：`encode → bytes → decode → receiver`。仅 fake-quant 或直接读取内存中的浮点 tensor 不能作为 packet 方法的最终结果。

## 10. 论文中可采用的方法组织

建议主论文不把观察、问题设定和方法混写：

```text
3. Problem Setting and Motivating Analysis
   3.1 Heterogeneous KV communication setting and cost definition
   3.2 Channel redundancy / latent-bottleneck diagnostic
   3.3 Sequence-frequency diagnostic
   3.4 Group-wise quantization sensitivity

4. Method
   4.1 Overview: learnable communication codec
   4.2 K/V-joint latent cache transform
   4.3 Group-wise frequency-domain quantization
   4.4 Local-global discrete rate allocator
   4.5 Rate-utility training and packetized inference
```

第 3 节的每一个 observation 都需要匹配实验图或诊断：例如 spectrum energy curve、不同 group 的量化敏感度、latent 宽度–性能–bytes 曲线。第 4 节只解释如何由该 observation 导出模块设计。

## 11. 当前需要确认或完成的事项

### 11.1 在写代码前需明确的接口

- [x] 层路由：从 source/receiver 的 terminal layer 开始向前逐层对齐，一个 receiver layer 对应一个 source layer。
- [x] encoder/decoder 的跨层参数：默认采用 LCF-style 的 per-layer 参数；“shared”仅指同一层内 K/V 共用 encoder 结构。跨层共享保留为参数效率消融，不再作为默认推荐。
- [x] decoder 规范：每层采用两个不共享参数的分支，`Decoder_K, Decoder_V: 64→256→C_r`，其中 `256=4×64`。
- [x] 初版实际 entropy coder：使用 zlib；后续可替换为专用熵模型或其他 coder。
- [x] `s_g`：每个样本、每个 `layer×K/V` group 在 DCT 系数的 sequence×latent-channel 维度上计算 RMS，使用 FP32 并 stop-gradient；adaptive packet 当前以 little-endian FP32 传输 scale。
- [x] 量化步长约定：`Δ[g,f] = alpha[g] × q_base[f]`，其中 `q_base` 从 `q_base_min` 按 `q_base_power` 增长到 `q_base_max`；alpha 越大表示步长越粗。
- [x] RoPE/position 协议：sharer K 去 RoPE、receiver K 按通信 prefix 的 receiver position 重新施加 RoPE；Value 不施加 RoPE，并统一 position IDs、attention mask 与 prefix offset。

### 11.2 最小可行验证顺序

1. **无量化 codec：** LCF bottleneck + decoder 是否能让 receiver 使用重建 KV。
2. **DCT identity control：** DCT/IDCT 但不量化，确认变换本身不造成明显损失。
3. **固定量化 QAT：** 同一训练预算下比较统一固定 alpha 与 group-wise alpha，避免把“训练内量化”误归因给 allocator。
4. **真实 packet round trip：** 报告 accuracy、实际 bytes、metadata 开销和 CR（分母为 raw sharer KV）。
5. **rate–utility sweep：** 多个 `λ`、alpha candidates，至少多 seed；不预设真实 rate 对 `λ` 单调。
6. **模块消融：** no-LCF / latent width、no-DCT、uniform alpha、local-only、no-global-context、不同 scale 策略。
7. **系统与泛化：** 更多 model pairs、任务、context length；有真实网络条件后再报告端到端 latency。

当前主实验配置为 Qwen3-0.6B receiver、可替换 sharer、OpenHermes 500k、LCFProjectedKV、两阶段 raw→QAT 训练。adaptive eval 使用同一 QAT allocator checkpoint 和 adaptive packet；MMLU-Redux、OpenBookQA、ARC-Challenge、C-Eval 应分别使用同一 packet codec 进行评测。

### 11.3 目前应避免写入摘要/结论的表述

- “显著降低端到端时延”（需要真实带宽、编码/解码与网络测量）；
- “自适应 allocator 找到最优或因果的重要性”；
- “低频就是关键信息，高频都是噪声”；
- “比所有 T2T/cache communication 方法更高效”；
- 将旧原型的 `mapped-BF16` 压缩率当作 LCF-first 对 raw sharer KV 的结果。

## 12. 一句话版本

> 对 sharer 的 K/V 先进行联合、可学习的 channel bottleneck，再将 K/V latent 分别变换到序列频域，并以 receiver 下游 CE 与可导 rate proxy 联合训练每个 `layer×K/V` group 的离散量化强度；部署时将真实熵编码 packet 解码为 receiver-compatible prefix KV，以评测准确率–payload 权衡。
