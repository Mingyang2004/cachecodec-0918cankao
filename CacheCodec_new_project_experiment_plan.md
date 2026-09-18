# CacheCodec 新版项目：实验与训练方案定稿

> **版本定位**：从“提出一种新的 cache collaboration 方法”转向“提出一个可插入不同 KV-cache collaboration operator 的通信压缩框架”。  
> **核心目标**：在 **Concat/Prefix** 与 **C2C Fusion** 两类不同的 cache-consumption 方式下，使用同一套 JCB + DCT + Quantization + zlib 通信框架，尽可能减少实际传输 payload 和端到端延迟，同时尽量保持原始 collaboration accuracy。  
> **统一评测原则**：所有 MCQ 方法一律使用相同的 **greedy decode -> parser -> option label -> accuracy** 流程，不使用 logit-based MCQ scoring。

---

# 0. 一句话定义新版论文

**CacheCodec 是位于 Sharer 与 Receiver 之间的通信编码框架，而不是新的 collaboration operator。**

需要证明的是：

\[
\text{Full-KV collaboration}
\approx
\text{Compressed-KV collaboration}
\]

同时：

\[
\text{actual wire bytes}\downarrow,\qquad
\text{TTFT}\downarrow,\qquad
T_{\rm E2E}\downarrow.
\]

并且这一结论同时成立于：

1. **Concat / Prefix cache communication**
2. **C2C cache fusion**

因此论文的重点不再是：

> “我们的 Concat accuracy 是否高于 C2C？”

而是：

> “无论 Receiver 最后通过 prefix 还是 fusion 消费 Sharer cache，我们都能压缩这条通信链路，并保持其原有 utility。”

---

# 1. 论文核心 Research Questions

## RQ1：Operator Generality

同一套通信 codec 思路能否同时应用于：

\[
\text{Concat/Prefix}
\]

和：

\[
\text{C2C Fusion}
\]

而不要求重新设计 Receiver LLM？

## RQ2：Compression-Utility Trade-off

在固定 downstream collaboration operator 时，是否能够减少真实 transmitted bytes，同时保持：

\[
Acc_{\rm compressed}
\approx
Acc_{\rm full-cache}?
\]

## RQ3：DCT 是否真的有用

相对于直接在 JCB latent 上量化：

\[
Z\rightarrow Q(Z),
\]

是否

\[
Z\rightarrow DCT(Z)\rightarrow Q
\]

能在 **相同 actual bitrate** 下获得更高 accuracy？

## RQ4：QAT 是否优于 PTQ

测试时直接加入 transform quantization（PTQ）和在训练 forward 中加入 fake quantization（QAT）相比，QAT 是否能够显著提高高压缩率下的 accuracy retention？

## RQ5：系统收益

考虑：

- JCB encode/decode
- DCT/IDCT
- quant/dequant
- GPU <-> CPU copy
- serialization
- zlib6
- actual network transfer
- collaboration operator
- Receiver first-token / answer decode

之后，CacheCodec 是否仍能降低 TTFT 和 E2E？并且在什么 bandwidth 下达到 break-even？

---

# 2. 需要明确的论文 Claim

论文应使用：

> **operator-agnostic / plug-compatible communication codec**

而不要轻易写：

> “同一套权重完全不训练即可插入任意 operator。”

当前设计里：

- JCB encoder / codec core 的思想和结构统一；
- Concat 与 C2C 的 reconstruction interface 不同；
- 两个分支允许重新训练自己的 JCB decoder；
- C2C 原始 projector/fuser 结构不修改。

因此最准确的 claim 是：

> **The same communication-coding framework is compatible with both prefix-based and fusion-based cache collaboration, while preserving the original downstream consumption mechanism.**

若后续“冻结已有 C2C，只训练 JCB/codec”实验成功，可以进一步称为：

> **drop-in adaptation**

但主论文不要在实验前过度承诺 zero-training plug-in。

---

# 3. 总体通信结构

完整通信 codec：

\[
C_S
\rightarrow
E_{\rm JCB}
\rightarrow
Z
\rightarrow
DCT
\rightarrow
Q
\rightarrow
\text{serialization}
\rightarrow
\text{zlib6}
\rightarrow
\text{network}
\rightarrow
\text{unzlib}
\rightarrow
Q^{-1}
\rightarrow
IDCT
\rightarrow
D_{\rm JCB}.
\]

其中：

- **JCB bottleneck**：有损，压缩 head/channel/KV 维度；
- **DCT**：可逆，不丢信息；
- **Quantization**：有损；
- **zlib6**：无损，仅影响实际 packet 大小和 CPU latency。

训练时：

- zlib **不进入计算图**；
- PTQ 训练时不加入 quantization；
- QAT 训练时加入 DCT + fake quant + IDCT；
- 主训练 loss 保持 **response-token CE loss**；
- 不再使用旧版的 learned group-wise \(lpha_g\)；
- 不再使用 differentiable rate loss 作为主方法。

---

# 4. JCB：Joint Cache Bottleneck

> JCB 来源于 joint K/V latent bottleneck 的思想，但论文中不要再把自己的模块命名为 LCF，避免和 **Latent Cache Flow (LCF)** 混淆。

定义：

\[
F_S = H_S D_S,
\qquad
F_R = H_R D_R.
\]

默认：

\[
R=128,
\qquad
R_K=R_V=64.
\]

## 4.1 每层输入

Sharer 某层：

\[
K_S,V_S
\in
\mathbb R^{B\times H_S\times N\times D_S}.
\]

flatten：

\[
K_S:
[B,H_S,N,D_S]
\rightarrow
[B,N,F_S],
\]

\[
V_S:
[B,H_S,N,D_S]
\rightarrow
[B,N,F_S].
\]

K/V concat：

\[
X_S=[K_S;V_S]
\in
\mathbb R^{B\times N\times 2F_S}.
\]

### 默认参数策略

- 每个 transmitted / mapped layer 使用一套独立 JCB 参数；
- 同一层内 K/V 共用 encoder trunk；
- K latent 与 V latent 使用独立 output heads；
- decoder 的 K/V branch 独立；
- 默认不做 cross-layer weight sharing；
- 若后续需要压参数量，再把 layer sharing 做成 ablation，而不是主方法。

## 4.2 JCB Encoder：Concat 与 C2C 共用同一结构

对于每个 token 独立执行：

\[
2F_S
\rightarrow
R
\rightarrow
4R
\rightarrow
R.
\]

默认 \(R=128\)：

```text
Linear(2*Fs, 128, bias=True)
GELU
Linear(128, 512, bias=True)
GELU
Linear(512, 128, bias=True)
```

得到：

\[
H\in\mathbb R^{B\times N\times128}.
\]

之后：

```text
K latent head:
Linear(128, 64, bias=True)

V latent head:
Linear(128, 64, bias=True)
```

得到：

\[
Z_K,Z_V
\in
\mathbb R^{B\times N\times64}.
\]

默认：

- activation = GELU
- dropout = 0
- 不额外加入 LayerNorm
- 不加入 residual
- latent output 不加 activation

---

# 5. Concat / Prefix Pipeline

## 5.1 Pure Concat baseline

Pure Concat 保持现有简单 Sharer -> Receiver projector / MLP，不加入 JCB / DCT / quantization。

抽象流程：

\[
C_S
\rightarrow
P_{S\rightarrow R}
\rightarrow
\widetilde C_R
\rightarrow
\text{Receiver prefix}.
\]

其中 K 必须处理 RoPE。

## 5.2 Concat 的 RoPE 规则

Sharer K 原始 cache 已包含 sender RoPE：

\[
K_S^{rope}.
\]

Concat 分支在进入 JCB / projector 前：

\[
K_S^{rope}
\rightarrow
K_S^{pre-rope}.
\]

JCB / codec 操作：

\[
(K_S^{pre-rope},V_S)
\rightarrow
\widetilde K_R^{pre-rope},\widetilde V_R.
\]

Receiver 端再应用 Receiver RoPE：

\[
\widetilde K_R^{pre-rope}
\rightarrow
RoPE_R
\rightarrow
\widetilde K_R^{rope}.
\]

因此：

\[
\boxed{
K_S^{rope}
\rightarrow
K_S^{pre-rope}
\rightarrow
JCB/Codec
\rightarrow
K_R^{pre-rope}
\rightarrow
RoPE_R
}
\]

V 不做 RoPE strip/restore。

### Position ID

- prefix KV 使用 prefix 自身对应的 Receiver position IDs；
- Receiver local prompt 的 position IDs 从 prefix length 后继续；
- 不允许对已经做过 RoPE 的 K 再做一次 RoPE；
- 保持当前 Concat 实现的位置语义不变。

## 5.3 Concat 不做 tokenizer alignment

Concat 的 transferred cache 是一个独立 virtual prefix。

因此：

- source token sequence 不需要和 Receiver local tokens一一对齐；
- 不需要 C2C 那种 tokenizer alignment；
- sequence length \(N_S\) 可以作为 prefix length 保留下来。

## 5.4 Concat + JCB Decoder

Concat 下，JCB decoder 直接输出 Receiver-compatible cache：

\[
F_{\rm out}=F_R=H_R D_R.
\]

### K decoder

\[
64
\rightarrow
256
\rightarrow
F_R.
\]

```text
Linear(64, 256, bias=True)
GELU
Linear(256, Hr*Dr, bias=True)
```

### V decoder

```text
Linear(64, 256, bias=True)
GELU
Linear(256, Hr*Dr, bias=True)
```

输出：

\[
\widetilde K_R^{pre-rope},
\widetilde V_R
\in
\mathbb R^{B\times H_R\times N_S\times D_R}.
\]

K 再恢复 Receiver RoPE。

### 最终 Concat + JCB

\[
C_S^{pre-rope}
\rightarrow
JCB_E
\rightarrow
Z
\rightarrow
JCB_D^{concat}
\rightarrow
\widetilde C_R^{pre-rope}
\rightarrow
RoPE_R
\rightarrow
C_R^{prefix}.
\]

### Concat + Ours

\[
C_S^{pre-rope}
\rightarrow
JCB_E
\rightarrow
Z
\rightarrow
DCT+Q+zlib
\rightarrow
network
\rightarrow
unzlib+Q^{-1}+IDCT
\rightarrow
JCB_D^{concat}
\rightarrow
C_R^{prefix}.
\]

---

# 6. C2C Fusion Pipeline

## 6.1 原始 C2C 不改 projector/fuser

C2C 原始逻辑继续保持：

\[
(C_S,C_R)
\rightarrow
F_{\rm C2C}
\rightarrow
C_R^{fused}.
\]

需要保留原 repo 的：

- layer mapping
- tokenizer alignment（当该 model pair 需要时）
- source/receiver flatten
- feature concat
- projector
- fusion
- gate
- Receiver residual

这些逻辑都不因为 CacheCodec 改动。

## 6.2 C2C 分支不做 pre-RoPE / re-RoPE

C2C+JCB 接收到什么 source KV，就重建同一语义接口的 source KV。

因此：

\[
C_S
\rightarrow
JCB
\rightarrow
\hat C_S
\rightarrow
\text{original C2C}.
\]

不额外：

- strip Sharer RoPE
- restore Receiver RoPE

保持原始 C2C 对 source cache 的处理方式。

## 6.3 C2C Tokenizer Alignment

顺序固定为：

\[
C_S
\rightarrow
JCB/Codec
\rightarrow
\hat C_S
\rightarrow
\text{original C2C tokenizer alignment}
\rightarrow
F_{\rm C2C}.
\]

也就是说：

- 网络上传输的是未做 target-tokenizer alignment 的 compressed Sharer message；
- Receiver 端 decode 回 Sharer-layout KV；
- 后续是否 alignment 完全继承 C2C repo 的原逻辑；
- 若某个 pair 原 repo 配置 `alignment=False`，保持 False；
- 不额外强制 alignment。

## 6.4 C2C + JCB Decoder

C2C 下必须恢复 Sharer layout：

\[
F_{\rm out}=F_S=H_S D_S.
\]

### K decoder

\[
64
\rightarrow
256
\rightarrow
F_S.
\]

```text
Linear(64, 256, bias=True)
GELU
Linear(256, Hs*Ds, bias=True)
```

### V decoder

```text
Linear(64, 256, bias=True)
GELU
Linear(256, Hs*Ds, bias=True)
```

输出：

\[
\hat K_S,\hat V_S
\in
\mathbb R^{B\times H_S\times N_S\times D_S}.
\]

之后：

\[
(\hat C_S,C_R)
\rightarrow
\text{original C2C projector/fuser}.
\]

---

# 7. 两个 JCB branch 的统一对照

| Item | Concat / Prefix | C2C Fusion |
|---|---|---|
| Encoder input | \(2H_SD_S\) | \(2H_SD_S\) |
| Encoder | \(2F_S\to128\to512\to128\) | 相同 |
| K latent | 64 | 64 |
| V latent | 64 | 64 |
| DCT/Q/zlib | 相同 | 相同 |
| Decoder K | \(64\to256\to H_RD_R\) | \(64\to256\to H_SD_S\) |
| Decoder V | \(64\to256\to H_RD_R\) | \(64\to256\to H_SD_S\) |
| RoPE | strip Sharer K -> restore Receiver K | 不额外操作 |
| Tokenizer align | 不需要 | 保持原 C2C logic |
| 后续 projector | JCB decoder 已完成 S->R | 原始 C2C projector 完全保留 |
| Receiver usage | prefix | fusion |

---

# 8. Frequency Codec

## 8.1 DCT

对于每层、每个 K/V latent：

\[
Z_t^l\in\mathbb R^{B\times N\times64},
\quad
t\in\{K,V\},
\]

沿 **sequence dimension** 做 1-D DCT：

\[
U_t^l=DCT_N(Z_t^l).
\]

DCT：

- 不降低 sequence length；
- 不删除 coefficient；
- 本身无损；
- 目的仅是 decorrelation / energy compaction；
- 同一实现用于 Concat 和 C2C。

主版本先使用 full-sequence DCT，不在主方法中加入 block-DCT，以避免额外设计变量。

---

# 9. Nonlinear Shared Quantization Table

## 9.1 默认 quant table

所有：

- layer
- K/V
- latent channel

使用同一套 frequency-dependent quant table。

quant table 的最小值和最大值固定为：

\[
q_{\min}=1,
\qquad
q_{\max}=8.
\]

默认使用 **非线性 power-law table**：

\[
r_f=\frac{f}{N-1},
\]

\[
\boxed{
q_f
=
1+7r_f^2
}
\]

其中：

\[
f=0,\dots,N-1.
\]

因此：

- low-frequency coefficient 使用更小 step；
- high-frequency coefficient 使用更大 step；
- table 单调非线性增长；
- 所有 cache groups 完全共用。

这就是项目中所说的：

> `quant_table = [1, 8]`

其中 `[1,8]` 表示端点范围，不是 literal 只有两个量化值。

## 9.2 Global Quality Multiplier

为了形成 rate-accuracy curve，引入一个 **全局** quality multiplier：

\[
\beta>0.
\]

实际 quantization step：

\[
\Delta_f=\beta q_f.
\]

量化：

\[
Q[f,c]
=
round
\left(
\frac{U[f,c]}
{\beta q_f}
\right).
\]

反量化：

\[
\hat U[f,c]
=
Q[f,c]\beta q_f.
\]

注意：

\[
\beta
\]

对所有 layer / K/V / channel / sample 都相同。

因此这不是 adaptive bit allocation，也不是 learned importance。

建议 PTQ sweep：

\[
\beta\in\{0.25,0.5,1,2,4\}.
\]

默认 Main Table 使用：

\[
\beta=1.
\]

最终 beta 不允许在 test benchmark 上单独调；如需选单一 operating point，只能用 held-out OpenHermes validation 选择。

---

# 10. Scale：作为独立 Evaluation Ablation

主训练：

\[
\boxed{\text{Scale OFF}}
\]

先不把 scale 放进训练。

## 10.1 No-scale

\[
Q=round(U/\Delta).
\]

## 10.2 Scale-on

评测时可选：

\[
s_{b,l,t}
=
\sqrt{
\frac{1}{N\cdot64}
\sum_{f,c}
U_{b,l,t,f,c}^2+\epsilon
}.
\]

normalize：

\[
\widetilde U=U/s.
\]

量化：

\[
Q=
round
\left(
\frac{\widetilde U}{\beta q_f}
\right).
\]

重建：

\[
\hat U=s\beta q_fQ.
\]

Scale 只解决 dynamic-range mismatch，不代表 importance。

若 scale 打开：

- scale 必须计入 packet metadata；
- 建议 fp16 存储；
- actual wire bytes 必须包含 scale overhead。

Main method 先使用 no-scale；scale-on 作为 ablation。

---

# 11. Quantized Integer 与 Packet

## 11.1 Integer dtype

优先使用 `int16`，但只有在所有评测 operating points 上确认 overflow count = 0 时才使用。

若发生 overflow：

- 不允许 silently clip；
- 改用 int32；
- 将最终 dtype 写进实验设置；
- 实际 packet bytes 以真实 dtype 为准。

## 11.2 zlib

固定：

```text
zlib level = 6
```

流程：

\[
Q
\rightarrow
serialize
\rightarrow
zlib6
\rightarrow
packet.
\]

Receiver：

\[
packet
\rightarrow
unzlib
\rightarrow
deserialize
\rightarrow
Q.
\]

zlib：

- 无损；
- 不进入训练图；
- 不影响 accuracy；
- 会影响 packet bytes；
- 会增加 CPU encode/decode latency。

## 11.3 Packet size 定义

主论文中的：

\[
S_{\rm packet}
\]

必须使用 **zlib6 最终输出的真实 byte 数**，并包含所有需要随 sample 传输的 metadata，例如：

- variable shape / sequence length（若协议中不是预共享）
- beta / quality ID（若不是系统预设）
- scale（Scale-on 时）
- dtype / packet header（若需要）

模型配置、固定 qtable、固定 codec version 等可认为 sender/receiver 预共享，不需要每个 sample 重复发送。

---

# 12. Compression Ratio

主 compression ratio：

\[
\boxed{
CR
=
\frac{
S_{\rm FullSharerKV}^{bf16}
}{
S_{\rm packet}
}
}
\]

其中 full Sharer KV bytes：

\[
S_{\rm FullSharerKV}^{bf16}
=
2
\times
L_{\rm tx}
\times
N
\times
H_S
\times
D_S
\times
2\text{ bytes}.
\]

第一个 2 表示 K/V。

另外可以辅助报告 JCB latent ratio，但主通信指标必须是：

\[
CR_{\rm wire}.
\]

---

# 13. 训练数据与训练公平性

## 13.1 统一训练数据

主实验全部使用：

> **OpenHermes-2.5 first 500K**

保持与当前 C2C training framework 一致。

建议：

- 99% train
- 1% held-out validation
- max sequence length = 2048
- response-only next-token CE
- Sharer / Receiver base LLM frozen

任何 downstream MCQ benchmark：

- 不参与训练；
- 不用于调 beta；
- 不用于 checkpoint selection。

## 13.2 统一训练 objective

所有 trainable collaboration / compression variant：

\[
\boxed{
L=L_{\rm CE}
}
\]

其中只对 assistant response token 计算 causal next-token CE。

不重新引入：

- rate loss
- learned entropy loss
- group-wise alpha loss
- downstream benchmark-specific CoT

这样能最大限度减少 confound。

---

# 14. 训练 Schedule

主实验保持 C2C 原训练预算：

\[
\boxed{
1\text{ epoch}
\approx
1929\text{ optimizer steps}
}
\]

目标是让以下方法训练预算一致：

- Concat
- Concat + JCB
- Concat + QAT
- C2C
- C2C + JCB
- C2C + QAT

推荐保留当前 C2C 配置中的：

- effective batch size = 256
- learning rate = \(1\times10^{-4}\)
- linear LR schedule
- 10% warmup

其余 optimizer 细节：

- AdamW betas
- weight decay
- grad clipping
- precision mode
- gradient accumulation

**完全沿用当前 C2C training config，不在不同方法间修改。**

论文最终 appendix 必须把这些值从实际 config 原样抄出。

---

# 15. 三类训练方式

## 15.1 Baseline

### Concat

训练现有 simple projector / translator：

\[
C_S\rightarrow \widetilde C_R.
\]

Loss：

\[
L_{\rm CE}.
\]

### C2C

训练原始 C2C projector/fuser。

Loss：

\[
L_{\rm CE}.
\]

## 15.2 +JCB

无 DCT / quantization。

### Concat + JCB

训练：

- JCB encoder
- Concat-specific JCB decoder

不再使用 pure-Concat simple projector。

### C2C + JCB

训练：

- JCB encoder
- source-layout JCB decoder
- 原始 C2C projector/fuser

Main setting 推荐 joint train，因为它能公平比较完整 pipeline。

同时额外做一个 strict plug-in control：

> load pretrained C2C -> freeze C2C -> only train JCB

用来证明 codec/JCB 可以适配一个已有 C2C operator。

## 15.3 PTQ

先获得已经训练好的 Concat+JCB 或 C2C+JCB。

不再训练。

仅 evaluation forward 中插入：

\[
DCT
\rightarrow
Q
\rightarrow
Q^{-1}
\rightarrow
IDCT.
\]

因此：

\[
\boxed{
\text{PTQ = zero extra optimizer step}
}
\]

## 15.4 QAT

QAT 总 optimizer step 仍固定为：

\[
1929.
\]

建议：

### Step 0 - warmup end

前 10%：

\[
\approx193\text{ steps}
\]

不加入 fake quant。

### Warmup 后

开启：

\[
DCT
\rightarrow
FakeQuant_{\rm STE}
\rightarrow
IDCT.
\]

总训练 budget 不增加。

Fake quant：

\[
Q=round(U/\Delta),
\]

backward 使用 STE。

zlib 不进入训练。

Main QAT operating point：

\[
\beta=1.
\]

若算力允许，额外训练：

\[
\beta\in\{0.5,1,2\}
\]

用于 QAT Pareto curve。

若算力有限：

- PTQ sweep 全 beta；
- QAT 只做 beta=1 主点。

---

# 16. 统一 Evaluation Protocol

## 16.1 所有方法必须使用同一个 MCQ evaluator

包括：

- Receiver-only
- Sharer-only
- T2T
- Concat
- Concat+JCB
- Concat+Ours
- C2C
- C2C+JCB
- C2C+Ours

全部：

\[
\boxed{
greedy\ decode
\rightarrow
parser
\rightarrow
option\ label
\rightarrow
accuracy
}
\]

绝对不要混用：

- direct logits MCQ
- likelihood ranking
- free-form judge

## 16.2 Generation 设置

必须固定现有 C2C evaluator 的：

- prompt template
- answer instruction
- `do_sample=False`
- beam setting
- max_new_tokens
- stopping criteria
- parser regex / matching rule

所有方法完全相同。

建议：

- 保存所有 raw generation；
- 保存 parser 提取结果；
- parser failure 直接记为 incorrect；
- 单独报告 parser failure rate。

---

# 17. 主 Evaluation Datasets

为了和 C2C 原 framework 直接对齐，主 MCQ 建议固定：

1. OpenBookQA
2. ARC-Challenge
3. MMLU
4. C-Eval

Main Table 报：

- 每个 dataset accuracy
- Macro / weighted average（必须提前固定定义）
- parser failure rate

如需扩展：

- MMLU-Redux
- MMLU-Pro

放 appendix / robustness。

---

# 18. 主实验 Methods Matrix

建议 Main Table 至少包含：

| Method | Collaboration | Wire Representation |
|---|---|---|
| Receiver-only | none | none |
| Sharer-only | none | none |
| T2T | text | text bytes |
| Concat | prefix | Full BF16 Sharer KV |
| Concat + JCB | prefix | BF16 JCB latent |
| Concat + Codec-PTQ | prefix | DCT + quant + zlib packet |
| **Concat + Codec-QAT** | prefix | DCT + quant + zlib packet |
| C2C | fusion | Full BF16 Sharer KV |
| C2C + JCB | fusion | BF16 JCB latent |
| C2C + Codec-PTQ | fusion | DCT + quant + zlib packet |
| **C2C + Codec-QAT** | fusion | DCT + quant + zlib packet |

建议论文图表中不要使用 `+LCF`，统一使用 `+JCB`，避免与 Latent Cache Flow 混淆。

---

# 19. Main Table 1：Accuracy + Wire Cost

推荐列：

| Method | OBQA | ARC-C | MMLU | C-Eval | Avg | Payload KB | CR |
|---|---:|---:|---:|---:|---:|---:|---:|

主要比较：

\[
Acc_{\rm Concat+Ours}
\quad vs.\quad
Acc_{\rm Concat}
\]

以及：

\[
Acc_{\rm C2C+Ours}
\quad vs.\quad
Acc_{\rm C2C}.
\]

不要把 Concat+Ours 必须超过 C2C 作为论文成功条件。

---

# 20. Main Table 2：Systems

推荐：

| Method | Encode ms | Network ms | Decode ms | Operator ms | TTFT ms | E2E/TTEA ms | p95 E2E |
|---|---:|---:|---:|---:|---:|---:|---:|

至少给：

- Full Concat
- Concat + JCB
- Concat + Ours
- Full C2C
- C2C + JCB
- C2C + Ours
- T2T

---

# 21. 最重要的 Pareto Figures

## Figure A：Accuracy vs Actual Wire Bytes

横轴：

\[
S_{\rm packet}
\]

纵轴：

\[
Accuracy.
\]

分别画：

- Concat PTQ
- Concat QAT
- C2C PTQ
- C2C QAT

beta sweep：

\[
0.25,0.5,1,2,4.
\]

## Figure B：Accuracy vs E2E Latency

同样画：

- Concat
- Concat+Ours
- C2C
- C2C+Ours
- T2T

## Figure C：E2E Latency vs Bandwidth

横轴：Bandwidth  
纵轴：\(T_{\rm E2E}\)

展示：

- Full KV
- JCB
- Codec
- T2T

并标出 break-even bandwidth。

---

# 22. 必做 Ablation

## A1. JCB 是否有效

比较：

\[
FullKV
\]

vs.

\[
JCB.
\]

R sweep：

\[
R\in\{64,128,256\}.
\]

默认：

\[
R=128.
\]

## A2. DCT 是否必要

比较：

### Direct Quant

\[
Z\rightarrow Q\rightarrow zlib
\]

vs.

### DCT Quant

\[
Z\rightarrow DCT\rightarrow Q\rightarrow zlib.
\]

重点比较：

\[
Accuracy
\]

at matched actual packet bytes。

## A3. Linear vs Nonlinear Quant Table

默认：

\[
q_f=1+7r_f^2.
\]

比较：

### Linear

\[
q_f=1+7r_f
\]

### Nonlinear

\[
q_f=1+7r_f^2.
\]

如果 nonlinear 没有稳定收益，不要把“JPEG-like nonlinear table”作为强 claim。

## A4. PTQ vs QAT

同一 beta 比较：

\[
PTQ
\]

和：

\[
QAT.
\]

至少：

\[
\beta=1.
\]

## A5. Scale Off vs Scale On

只改变 scale，其余全部固定。

比较：

- accuracy
- actual bytes
- extra metadata
- encode/decode latency

Main method 先默认 Scale Off。

## A6. zlib Off vs On

量化 tensor 完全相同。

比较：

- uncompressed integer bytes
- zlib6 bytes
- compression time
- decompression time
- total E2E

Accuracy 应严格一致。

---

# 23. 推荐但不是 Main 必须的 Ablation

## B1. Source Intervention

对 C2C / Concat：

- real Sharer cache
- zero Sharer cache
- shuffled cache
- wrong-sample cache

用于诊断真正的 source dependence。

这一项不要放成主线，但可以解释 C2C 的 zero-cache 现象。

## B2. Pre-RoPE vs Post-RoPE for Concat

证明为什么 Concat 对 K 使用 pre-RoPE 处理。

## B3. Strict C2C Drop-in

1. 训练原始 C2C；
2. freeze C2C；
3. 加 JCB / codec；
4. 只训练 JCB。

若效果好，这是非常强的 plug-in 证据。

---

# 24. 端到端 Timing：必须严格执行的测量原则

## 24.1 第一原则

CUDA 是异步的。

因此：

\[
\boxed{
\text{GPU timing 不能简单用 Python } time.time()
}
\]

---

# 25. GPU Component Timing

对：

- Sharer prefill
- JCB encoder
- DCT
- quant
- dequant
- IDCT
- JCB decoder
- Concat mapping
- C2C projector/fuser
- Receiver prefill
- decode kernel

使用 CUDA events。

标准模式：

```python
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
out = module(inp)
end.record()

torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

每次正式 benchmark 前：

```python
torch.cuda.synchronize()
```

正式结束后：

```python
torch.cuda.synchronize()
```

---

# 26. CPU / Mixed GPU-CPU Timing

zlib 是 CPU operation。

以下使用：

```python
time.perf_counter_ns()
```

并在 GPU/CPU 边界显式 synchronize：

- D2H
- H2D
- serialization
- zlib.compress(level=6)
- zlib.decompress
- socket send/recv

---

# 27. Codec Timing 必须包含哪些部分

## Encode

\[
T_{\rm enc}
=
T_{\rm JCB-E}
+
T_{\rm DCT}
+
T_Q
+
T_{\rm D2H}
+
T_{\rm serialize}
+
T_{\rm zlib}.
\]

## Decode

\[
T_{\rm dec}
=
T_{\rm unzip}
+
T_{\rm deserialize}
+
T_{\rm H2D}
+
T_{Q^{-1}}
+
T_{\rm IDCT}
+
T_{\rm JCB-D}.
\]

不能遗漏 D2H/H2D。

---

# 28. Full-KV Baseline 也必须走同样的真实通信路径

为了公平：

Full KV baseline 不能只计算：

\[
S/B.
\]

必须尽量通过和 compressed method 相同的 transport stack：

\[
GPU
\rightarrow
D2H
\rightarrow
serialize
\rightarrow
network
\rightarrow
deserialize
\rightarrow
H2D.
\]

区别只是：

- Full KV 不做 JCB/DCT/Q/zlib；
- CacheCodec 做。

否则会高估或低估 codec 的真实系统收益。

---

# 29. Network Timing

优先做：

\[
\boxed{\text{actual network transfer}}
\]

而不是只报告：

\[
S_{\rm payload}/B.
\]

建议：

1. 两个独立 process；
2. Sharer / Receiver 分别绑定不同 device；
3. 固定 transport；
4. FullKV/JCB/Codec 全部使用相同 transport。

如果有跨机器条件：

- 实际测链路 bandwidth；
- 可使用 iperf3 做环境校准；
- 主结果使用真实 send/recv。

---

# 30. Bandwidth Sweep

推荐：

\[
B\in
\{0.1,1,10,25,100\}\text{ Gbps}.
\]

若可用：

- Linux `tc`
- network shaping
- 可控交换网络

直接限制 socket bandwidth。

若无法做真实 bandwidth shaping：

- 实测硬件点作为主结果；
- \(S/B\) bandwidth sweep 只能作为 modeled analysis，必须清楚标注 modeled，不和真实 E2E 混写。

---

# 31. 跨机器时钟问题

不同 host 直接做：

\[
t_{\rm recv}^{hostB}
-
t_{\rm send}^{hostA}
\]

不可靠，除非有高精度 PTP。

因此主 E2E 建议：

- 一个 controller 发请求；
- 最终 answer 返回 controller；
- 同一个 monotonic clock 测 request-level latency。

或者使用 PTP synchronized clocks 再做 one-way breakdown。

---

# 32. TTFT 定义

统一：

\[
t_0=
\text{tokenized input 已准备好，正式 request execution 开始}.
\]

\[
\boxed{
TTFT=
t_{\rm first\ Receiver\ answer\ token}
-
t_0
}
\]

TTFT 必须包含真实 critical path：

- Sharer prefill
- codec encode
- communication
- codec decode
- collaboration operator
- Receiver prefill（若该方法需要）
- first-token computation

---

# 33. E2E / TTEA 定义

MCQ 中推荐称：

\[
\boxed{
TTEA =
t_{\rm final\ parsable\ answer}
-
t_0
}
\]

也可以在正文写：

> end-to-end answer latency

但定义必须固定。

同时报告：

- generated token count
- parser failure

避免某个方法因为生成更短文本而看起来更快。

---

# 34. C2C 的并行 Prefill

如果 Sharer / Receiver 在不同 device：

原始 C2C 的 Sharer prefill 与 Receiver prefill 可以并行。

因此真实 E2E critical path 不能简单写成两者相加。

实际执行类似：

```text
Sharer:   [------- prefill -------][codec][network]
Receiver: [----- prefill -----]                [C2C fusion][decode]
```

主 E2E 必须直接测真实执行图。

组件时间可以分别报告，但：

\[
\sum_i T_i
\]

不一定等于 E2E。

---

# 35. Concat 的 Prefill

Concat prefix 对 Receiver attention 有因果依赖。

因此默认：

1. Sharer produces cache
2. codec
3. Receiver gets prefix
4. Receiver local prefill
5. decode

除非当前实现已经能保证位置和 attention 语义正确地提前运行 Receiver local prefill，否则不要为了 latency 人为做不正确 overlap。

---

# 36. Warmup 与统计方式

每个 timing configuration：

### Warmup

至少 30 次 request，不记录。

### Measurement

- dataset-level 可直接记录每个 sample；
- microbenchmark 每个 shape 至少 100 repeats。

报告：

\[
median
\]

和：

\[
p95.
\]

可选同时报告：

\[
mean\pm std.
\]

---

# 37. Controlled Length Benchmark

除了真实 benchmark prompt，建议额外做：

\[
N\in\{256,512,1024,2048\}
\]

的 controlled sequence-length system benchmark。

因为：

\[
S_{KV}\propto N
\]

这是 CacheCodec 系统收益最自然的 scaling dimension。

报告：

- wire bytes vs N
- encode/decode latency vs N
- E2E vs N
- break-even bandwidth vs N

---

# 38. T2T Timing

T2T 的 TTFT / E2E 必须包含：

1. Sharer prefill
2. Sharer autoregressive intermediate text generation
3. text serialization / network
4. Receiver tokenize
5. Receiver prefill
6. Receiver answer decode

不能只拿最终 text byte 数和 KV byte 数比较。

另外记录：

- intermediate message token count
- intermediate message bytes

---

# 39. Reviewer-Controlled Checks

## Check 1：是不是 JCB 自己提高了 accuracy？

看：

\[
Concat
\rightarrow
Concat+JCB
\]

以及：

\[
C2C
\rightarrow
C2C+JCB.
\]

因此主结果能分开：

- JCB effect
- quantization effect

## Check 2：DCT 是否只是装饰？

比较：

\[
JCB+Q
\]

vs.

\[
JCB+DCT+Q
\]

at matched actual packet size。

## Check 3：是不是只是 zlib 有效？

比较：

- raw integer bytes
- zlib6 bytes

Accuracy 一样。

## Check 4：是不是只在 Concat 上有效？

必须同时给：

- Concat
- C2C

## Check 5：是不是 evaluator 差异？

所有方法完全相同：

\[
decode\rightarrow parser\rightarrow label.
\]

## Check 6：latency 是不是理论估计？

主系统结果必须包含真实 measured E2E。

Modeled \(S/B\) 只能作为 bandwidth analysis。

---

# 40. 建议 Success Criteria

这些是项目内部目标，不要在论文结果出来前写成承诺。

## Accuracy

希望：

\[
|\Delta Acc|
\le 1\text{ pp}
\]

相对对应 full-cache operator。

若高压缩点为 1-2 pp 也可以接受，只要 Pareto 明显更好。

## Compression

优先目标：

\[
\ge 8\times
\]

真实 wire compression。

若达到 16x+ 且 accuracy drop 很小，则非常有竞争力。

## Systems

至少在带宽受限场景：

\[
1-10\text{ Gbps}
\]

希望：

\[
TTFT_{\rm codec}
<
TTFT_{\rm fullKV}
\]

以及：

\[
E2E_{\rm codec}
<
E2E_{\rm fullKV}.
\]

最终以真实结果为准。

---

# 41. 旧版项目 -> 新版项目：需要删除/修改的内容

## 41.1 删除主线

旧版：

> different layer/K/V groups have different task importance

以及：

\[
\alpha_g\in\mathcal A
\]

和：

\[
L=L_{\rm CE}+\lambda\hat R.
\]

新版不再把 group-aware rate allocation 作为核心。

因此删除：

- learned \(lpha_g\)
- Gumbel allocation
- differentiable rate objective
- group sensitivity -> learned bit allocation 的主 contribution

这些可以保留为 future work。

## 41.2 保留

- KV communication bottleneck motivation
- JCB / joint K-V bottleneck
- sequence DCT
- actual packet / zlib
- E2E latency equation

---

# 42. 旧稿 Method 需要怎么改

## 原 3.1 Problem Formulation

旧稿偏：

\[
C_S
\rightarrow
codec
\rightarrow
\hat C_R
\rightarrow
virtual\ prefix.
\]

新版改成：

\[
C_S
\rightarrow
codec\ core
\rightarrow
\hat C^{interface}
\rightarrow
\mathcal O,
\]

其中：

\[
\mathcal O\in
\{
\text{Concat/Prefix},
\text{C2C Fusion}
\}.
\]

## 原 3.3 Latent Channel Fusion

改名：

> **Joint Cache Bottleneck (JCB)**

并明确：

- LCF-inspired;
- 不是论文主要 novelty；
- 是后续 transform coding 的连续 bottleneck。

## 原 3.4 Frequency-Domain Cache Coding

保留 DCT，但删掉 adaptive group-specific importance 的叙述。

重点改为：

> DCT decorrelates the sequence axis before shared quantization.

## 原 3.5 Adaptive Quantization

整节替换为：

> **Shared Nonlinear Frequency Quantization and QAT**

内容：

1. nonlinear shared qtable
2. global beta
3. optional scale
4. PTQ
5. QAT
6. zlib6 packetization

---

# 43. 新版 Paper 大纲

## 1 Introduction

1. Multi-LLM collaboration 能交换 richer latent state；
2. KV communication 避免中间 text decode / re-prefill；
3. 但 full KV 变成新的 wire bottleneck；
4. 现有方法更多优化 representation alignment / fusion / internal bottleneck；
5. 它们并没有解决“不同 collaboration operator 下如何统一地降低真实 transmitted bytes”；
6. 提出 CacheCodec；
7. 在 Concat/Prefix 与 C2C Fusion 两种 operator 上验证；
8. 报告 accuracy / actual payload / CR / TTFT / E2E。

## 2 Related Work

### 2.1 Multi-LLM Communication
### 2.2 Latent Bottleneck / Cross-model Cache Mapping
### 2.3 KV Compression
### 2.4 Transform Coding

## 3 Problem Formulation

定义 Sharer message：

\[
C_S.
\]

定义：

\[
\mathcal O_{\rm prefix}
\]

和：

\[
\mathcal O_{\rm fusion}.
\]

目标：

\[
\min S_{\rm packet}
\]

subject to：

\[
Acc_{\rm compressed}
\approx
Acc_{\rm full}.
\]

同时关注：

\[
T_{\rm E2E}.
\]

## 4 CacheCodec

### 4.1 JCB Encoder
### 4.2 Prefix/Concat Interface
### 4.3 C2C Fusion Interface
### 4.4 Sequence Transform
### 4.5 Shared Nonlinear Quantization
### 4.6 PTQ and QAT
### 4.7 Packetization
### 4.8 Latency Model

## 5 Experiments

### 5.1 Setup
### 5.2 Main Results: Accuracy vs Payload
### 5.3 Generality Across Operators
### 5.4 Accuracy-Payload Pareto
### 5.5 End-to-End Systems Evaluation
### 5.6 Ablation
### 5.7 Analysis

## 6 Discussion
## 7 Conclusion

---

# 44. 论文 Contributions 建议写成 3 条

### Contribution 1

**Operator-compatible KV communication coding**

提出一个可同时用于 prefix-based 与 C2C fusion-based collaboration 的 communication-coding framework。

### Contribution 2

**Simple and reproducible transform coding**

Joint K/V bottleneck + sequence DCT + shared nonlinear quantization + lossless zlib packetization，将 dense continuous KV message 变成实际小 bitstream，并支持 PTQ / QAT。

### Contribution 3

**Real end-to-end evaluation**

不只报告 tensor reduction，而是统一报告：

- downstream accuracy
- actual transmitted bytes
- compression ratio
- TTFT
- E2E/TTEA
- bandwidth-dependent break-even

---

# 45. 推荐实验优先级

## P0：必须完成

1. Concat baseline reproduction
2. C2C baseline reproduction
3. Concat + JCB
4. C2C + JCB
5. Concat + PTQ
6. C2C + PTQ
7. Concat + QAT
8. C2C + QAT
9. actual packet bytes
10. accurate TTFT/E2E timing

## P1：核心 Ablation

1. Direct Q vs DCT+Q
2. PTQ vs QAT
3. linear vs nonlinear table
4. R=64/128/256
5. scale off/on
6. zlib off/on

## P2：增强 Reviewer 说服力

1. strict frozen-C2C plug-in
2. source zero/shuffle
3. extra model pair
4. extra long-context / source-dependent task
5. controlled N=256/512/1024/2048 latency scaling

---

# 46. 实验记录规范

每个 run 必须保存：

- git commit
- random seed
- training config
- model pair
- JCB R
- qtable formula
- beta
- scale on/off
- PTQ/QAT
- integer dtype
- zlib level
- dataset split
- raw generation
- parsed label
- parser failure
- payload bytes/sample
- component latency/sample
- TTFT/sample
- E2E/sample

所有主表结果必须能从逐 sample 日志重新聚合。

---

# 47. Random Seeds

算力允许：

\[
3\text{ seeds}
\]

用于：

- Concat
- Concat+Ours
- C2C
- C2C+Ours

其余 ablation 可先单 seed。

如果算力不足：

- 主表单 seed；
- 最重要的 compressed/full pair 补 3 seeds；
- 明确报告 variance。

---

# 48. 最终主实验应该回答的 5 个问题

### Q1：CacheCodec 是否能压缩 Concat？

看 Concat vs Concat+Ours。

### Q2：是否也能压缩更强的 C2C？

看 C2C vs C2C+Ours。

### Q3：压缩是否真来自 DCT + quant + zlib，而不是只换了一个更好的 projector？

看 +JCB、DirectQ、DCTQ、zlib ablation。

### Q4：PTQ 就够了，还是 QAT 必须？

看 PTQ vs QAT。

### Q5：减少 bytes 是否真的转化成系统收益？

看 actual network TTFT / E2E / bandwidth sweep。

---

# 49. 项目最终技术主线

整篇论文不要围绕：

> “我们的量化器比别人更复杂。”

而要围绕：

\[
\boxed{
\text{Collaboration operator}
\perp
\text{Communication codec}
}
\]

也就是：

> **如何使用 KV 是一个问题，如何把 KV 低成本传过去是另一个问题。**

Concat 与 C2C 分别代表两种不同的 KV consumption mechanism：

\[
\text{Prefix conditioning}
\]

与：

\[
\text{Feature fusion}.
\]

CacheCodec 的价值是：

> 对这两种机制，都可以把 full Sharer cache 变成更小的实际网络 message，并尽量保持其原有 downstream utility。

这应当成为新版项目从 Introduction 到 Conclusion 始终不变的中心线。
