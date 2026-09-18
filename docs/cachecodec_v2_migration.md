# CacheCodec v2 migration

新版主线将 codec 与 collaboration operator 分离：

```text
Sharer KV -> JCB -> [B, S, 64] K/V latents
          -> 1-D DCT(sequence) -> fixed q_f=1+7r^2, beta
          -> dense int16 -> zlib level 6 -> transport
          -> zlib -> int16 -> dequant -> IDCT -> JCB decoder
```

`JCBProjector` 位于 `rosetta/model/jcb.py`，结构固定为 shared
`2*Fs -> 128 -> 512 -> 128`，再由独立 K/V heads 输出 64 维 latent；Concat
分支的 decoder 输出 Receiver geometry。`JCBC2CProjector` 另外提供只恢复
Sharer geometry 的 decoder 接口，供原始 C2C projector/fuser 前的通信适配使用。

Concat 评测可直接使用：

```text
recipe/train_recipe/cachecodec_concat_jcb_external.json
recipe/eval_recipe/cachecodec_concat_jcb_external.yaml
```

将 eval recipe 的 `codec.method` 设为 `jcb_dct_int16` 后，wrapper 会对每个
mapped layer 的 pseudo K/V latent 独立执行 DCT、固定量化、int16 overflow 检查、
zlib6 和重建；packet bytes 使用序列化后的真实字节数。该路径不会读取或创建
adaptive quant table，也不会产生 rate loss。

C2C 的原始 projector/fuser 仍保持不变；`JCBC2CProjector.decode_source` 的输出
接口是 Sharer `[B,Hs,S,Ds]`，训练和评测 wrapper 会在原始 projector/fuser 前调用
它，后续 tokenizer alignment、flatten+concat、projector、residual fusion 继续走
C2C 原逻辑。

## 外置运行目录

```bash
export CACHECODEC_MODEL_ROOT=/path/to/models
export CACHECODEC_DATA_ROOT=/path/to/data
export CACHECODEC_RUN_ROOT=/path/to/runs
```

训练和评测入口会递归展开 `${CACHECODEC_*}`。仓库根目录 `.gitignore` 忽略
checkpoint、数据和结果文件；因此 GitHub 只保存代码、recipe 模板和文档。

## 训练数据和控制

六个新版训练 recipe 固定使用 `teknium/OpenHermes-2.5` 的训练 split，
`num_samples=500000`、`max_word_count=2048`、`train_ratio=0.99`。99% 用于训练，
1% 用于脚本内部 CE 验证；MMLU、ARC、OpenBookQA 和 C-Eval 不参与训练或 checkpoint
选择。

基础训练预算沿用 C2C：1 epoch、`learning_rate=1e-4`、weight decay `0.01`、
linear scheduler、warmup ratio `0.1`，Sharer/Receiver base LLM 冻结，只更新
collaboration/projector/codec 参数。Concat 使用 per-device batch `1`、gradient
accumulation `16`；C2C 使用 per-device batch `4`、gradient accumulation `8`。

新版 recipe 设置 `save_steps=100` 和 `eval_steps=100`。当
`keep_only_latest_checkpoint=true` 时，每次写入 `checkpoint-N` 前删除同一 run
目录下旧的 `checkpoint-*`，但保留最终 `final/`。Periodic checkpoint 保存 projector、
`cache_codec_N`（启用时）、mapping 以及 optimizer/scheduler 状态。

可选提前停止：

```json
"early_stopping": {
  "enabled": true,
  "window_size": 20,
  "patience": 3,
  "min_delta": 0.001
}
```

脚本按 optimizer-step 平均训练 loss 更新滑动窗口；窗口跨度不超过 `min_delta`
连续 `patience` 次即停止，随后仍写出 `final/`。分布式训练由 rank 0 决策并广播
停止信号。

## Prompt 和 token 流

MCQ 评测统一调用 `rosetta/utils/evaluate.py` 的 `build_prompt`。`use_cot=false`
时模板为：

```text
Accurately answer the following question:

{question}

Choices:
{A-D choices}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is
```

`use_cot=true` 时只替换 instruction，要求简短推理后以
`The correct answer is ...` 结束。四个短 benchmark 都使用此模板，生成固定为
greedy (`do_sample=false`)，再由 parser 提取 A/B/C/D。

训练不调用 MCQ `build_prompt`。OpenHermes 样本是 chat messages，最后一条 assistant
message 是监督目标，由各模型自己的 chat template 渲染。

Concat 训练/评测使用两个独立 token 流：

```text
receiver tokenizer: messages[:-1] + generation prompt（训练时接 assistant response）
sharer tokenizer:   messages[:-1] + generation prompt
```

不做 token-to-token alignment；Sharer cache 经 JCB 后输出 Receiver geometry，作为
prefix 使用。C2C/Fusion 在 `is_do_alignment=true` 时使用原 C2C `TokenAligner`，按
`alignment_strategy` 对齐并 padding 两侧逻辑序列；JCB 只压缩 Sharer cache，恢复
`Hs×Ds` 后原 C2C 再执行 `start:end`、flatten、channel concat、projector 和 residual
fusion。关闭该选项时，评测复用 C2C-clean 的单一 Receiver token stream 给两侧模型，
保证 section 长度一致；这不是 tokenizer alignment。

## Loss 边界

新版 Concat 和 C2C-JCB 的 loss 只有 assistant response token 的 causal CE，prompt
label 为 `-100`。DCT、量化和 zlib 只在评测 packet 路径执行，不产生 rate loss。
旧 adaptive quant 分支仍为历史兼容代码，但新版 recipe 明确关闭，`training_stage=jcb`
也会拒绝误开启 adaptive quant。

## 新旧代码开关对照

| 目标 | 新版开关/入口 | 旧版路径 |
|---|---|---|
| Raw Concat baseline | `cache_alignment=concat`, `concat_projector.type=raw` | `lcf_first` / `lcf_projected_kv` |
| Concat + JCB | `cache_alignment=concat`, `concat_projector.type=jcb` | `lcf_first` / `lcf_projected_kv` |
| 原始 C2C Fusion | `cache_alignment=fuser`, `fusion_type=original` | 同一原始 `C2CProjector` |
| Sharer-side JCB | `cache_codec.enabled=true`, `cache_codec.type=jcb` | 无 |
| 评测频域 codec | `codec.method=jcb_dct_int16` | `cachejpeg` / adaptive packet |
| Adaptive quant/rate loss | 新版必须关闭 | 仅旧 recipe 兼容 |
| Token alignment | C2C 可由 `is_do_alignment` 控制；Concat 禁用 | 旧 C2C 同名配置 |

训练时 `cache_codec` 只做可学习 JCB encode/decode；默认训练仍不执行 DCT/int16/zlib。
QAT 组通过 `model.dct_qat.enabled=true` 在 JCB latent 与 decoder 之间插入
`DCT -> fake round(STE) -> IDCT`，仍只有 response-token CE、没有 rate loss，也不做
真实 zlib。评测 packet 组则使用真实 DCT/int16/zlib round-trip；两者可分别消融。

## 每个样本的完整流程

### Concat

1. Evaluator 用 Receiver tokenizer 和 Sharer tokenizer 分别渲染同一 MCQ prompt。
2. Sharer prefill 得到每层 `[B,Hs,Ss,Ds]`，capture 后保留 pre-RoPE K。
3. 根据 terminal layer mapping，每个 Receiver layer 选择一个 Sharer layer。
4. JCB flatten K/V 为 `[B,Ss,2HsDs]`，得到 shared `[B,Ss,128]`，再得到 K/V `[B,Ss,64]`。
5. 每个 K/V latent 沿 `Ss` 做正交 DCT；步长为 `beta*(1+7*(f/(Ss-1))^2)`。
6. round 到 int16，overflow 直接报错，不 silently clip；序列化后 zlib level 6。
7. Receiver 解包、反量化、IDCT，JCB decoder 输出 `[B,Hr,Ss,Dr]`。
8. 对 decoder K 应用 Receiver RoPE，拼成 prefix，Receiver 使用 offset position IDs 完成
   prefill 和 greedy answer generation。

### C2C Fusion

1. 如果 `is_do_alignment=true`，TokenAligner 先让两侧逻辑 token 序列等长并生成 mask；
   false 时两侧分别使用原始 tokenizer 流。
2. Sharer prefill 得到原始 Sharer cache；JCB 输入仍是每层 `Hs×Ds` 的 K/V，而不是
   Receiver cache，也不在 JCB 内做 flatten+concat 到 Receiver。
3. JCB decoder 恢复 `[B,Hs,Ss,Ds]` 的 Sharer K/V；评测中这一步之前经过 DCT/int16/zlib
   packet round-trip。C2C 不额外执行 de-RoPE/re-RoPE，保留 C2C-clean 使用的 cache
   空间；只有 Concat prefix 分支对 Key 做 pre-RoPE capture 和 Receiver-side RoPE。
4. 完全复用 C2C-clean：按 `start:end` 取 source/receiver section，分别 flatten，沿
   channel concat 后送入原始 C2C projector。
5. Projector 输出 Receiver geometry 的 residual cache；parallel 模式从 clean Receiver
   cache 累积 delta，最后写回 Receiver cache，后续生成不变。

因此两条路线共用 JCB/频域 codec，但 JCB decoder 的目标 geometry、RoPE 规则、token
alignment 和后续 cache consumer 不同。

## 输出目录

例如：

```text
$CACHECODEC_RUN_ROOT/
  checkpoints/
    concat_jcb_qwen25_05_to_qwen3_06/final/
    fusion_lcf_qwen25_05_to_qwen3_06/final/
  results/
    concat_jcb/
    fusion_lcf/
```

仓库只保留 recipe 和代码。训练 checkpoint 内的 `projector_N.*` 是原 collaboration
projector，`cache_codec_N.*` 是 Sharer-side JCB；评测 loader 会分别加载两类权重。

## 六组主实验矩阵

本版本固定比较同一 model pair（Receiver `Qwen3-0.6B`、Sharer
`Qwen2.5-0.5B-Instruct`）的六组配置。所有组都使用 OpenHermes-2.5 500k、
response-token causal CE；Receiver/Sharer LLM 冻结，只训练下表中标记的
projector/JCB。`+quant` 不重新训练，也不加入 rate loss，而是复用对应 JCB
checkpoint，在评测时对 Sharer→Receiver packet 做固定 `[1,8]` DCT 量化。

| 组别 | 训练入口 | 评测入口 | 传输内容 | downstream consumer |
|---|---|---|---|---|
| `concat` | `recipe/train_recipe/cachecodec_concat_raw_external.json` | `recipe/eval_recipe/cachecodec_concat_raw_external.yaml` | 被路由 Sharer 层的完整 K/V，序列化后传输 | `DirectConcatProjector` → Receiver prefix |
| `concat+jcb` | `recipe/train_recipe/cachecodec_concat_jcb_external.json` | `recipe/eval_recipe/cachecodec_concat_jcb_external.yaml` | JCB K/V latent 浮点值 | JCB decoder → Receiver prefix |
| `concat+jcb+quant` | 复用 `concat+jcb` checkpoint | `recipe/eval_recipe/cachecodec_concat_jcb_quant_external.yaml` | JCB latent → DCT → `[1,8]` dense int16 → zlib6 | JCB decoder → Receiver prefix |
| `fusion` | `recipe/train_recipe/cachecodec_fusion_raw_external.json` | `recipe/eval_recipe/cachecodec_fusion_raw_external.yaml` | 完整 Sharer K/V，序列化后传输 | 原始 C2C projector/fuser/gate/residual |
| `fusion+lcf` | `recipe/train_recipe/cachecodec_fusion_lcf_external.json` | `recipe/eval_recipe/cachecodec_fusion_lcf_external.yaml` | Sharer-side JCB/LCF latent 浮点值 | 原始 C2C projector/fuser/gate/residual |
| `fusion+lcf+quant` | 复用 `fusion+lcf` checkpoint | `recipe/eval_recipe/cachecodec_fusion_lcf_quant_external.yaml` | JCB/LCF latent → DCT → `[1,8]` dense int16 → zlib6 | 原始 C2C projector/fuser/gate/residual |

`concat` 的 raw packet 不做 JCB、DCT 或量化，用于测量被路由 Sharer 层完整 K/V 的
accuracy、真实序列化字节数和端到端时间；未被 terminal mapping 选中的层不会重复
传输。`concat+jcb` 隔离 JCB bottleneck，
`concat+jcb+quant` 再增加固定频域量化误差。Fusion 三组的唯一区别同样只在
Sharer→Receiver 传输边界，C2C-clean 的 projector、fuser、gate、residual 写回
逻辑保持不变。

### 对齐协议

两条路线都使用 `mapping=last_aligned` 进行 terminal layer routing：每个
Receiver layer 选择一个 Sharer layer；这是网络层路由，不是 tokenizer 对齐。

| 路线 | tokenizer alignment | 新版 Qwen3-0.6B + Qwen2.5-0.5B 配置 |
|---|---|---|
| Concat | 禁用；Receiver/Sharer 是独立 virtual prefix 与 local prompt | 不读取 `is_do_alignment` |
| Fusion | 可选；开启时调用原 C2C `TokenAligner` | `is_do_alignment=false`, `alignment_strategy=first` |

原仓库的 `C2C_0.6+0.5.json` 是 `false/first`；`C2C_0.6_1.json` 是针对
Llama-3.2-1B 的 `true/longest`。因此不能把 `longest` 或 `true` 作为所有
Fusion pair 的默认值。若复现 Llama pair，只需在对应 recipe 同时设置
`is_do_alignment=true` 和 `alignment_strategy=longest`；JCB decoder 仍恢复
Sharer `[B,Hs,S,Ds]`，然后再进入原 C2C alignment 和 fusion。

### 关键逻辑位置

- 训练入口：`script/train/SFT_train.py`；模型构建、冻结、每 100 optimizer steps
  保存/删除旧 checkpoint、滑动窗口 early stopping 都在此。
- 训练 Concat forward：`rosetta/model/wrapper.py::_forward_concat_lcf_first`；
  根据 `DirectConcatProjector` 或 `JCBProjector` 选择 raw/direct 或 JCB 分支。
- 训练 Fusion forward：`rosetta/model/wrapper.py::forward`；缓存先经过可选
  `JCBC2CProjector`，随后执行原 C2C projector/fuser/gate/residual。
- Raw Concat decoder：`rosetta/model/projector.py::DirectConcatProjector` 和
  `rosetta/cachejpeg_rosetta/cache_aligner.py::RawConcatCacheAligner`。
- JCB：`rosetta/model/jcb.py`；Concat 输出 Receiver geometry，C2C 输出
  Sharer geometry。
- 固定 packet：`rosetta/cachejpeg/packet.py::DCTInt16PacketCodec`；沿 sequence
  轴 DCT，量化表从低频 `1` 平滑到高频 `8`，所有 layer/K/V 共用同一规则，
  overflow 直接报错，zlib level 6。
- 评测入口：`script/evaluation/unified_evaluator.py`；四个短 bench 都走 greedy
  generation、选项 parser 和 accuracy，并从 `last_codec_stats` 读取 payload
  bytes、compression factor、encode/decode/transport 时间。T2T 入口为
  `rosetta/baseline/t2t.py`，通过 `model_name: t2t` 选择，不加载 Rosetta checkpoint。

### 运行命令

先指定外部目录（这些目录不会进入 Git）：

```bash
export CACHECODEC_MODEL_ROOT=/path/to/models
export CACHECODEC_DATA_ROOT=/path/to/data
export CACHECODEC_RUN_ROOT=/path/to/runs
```

训练任意一组：

```bash
CUDA_VISIBLE_DEVICES=1 python script/train/SFT_train.py \
  --config recipe/train_recipe/cachecodec_concat_external.json
```

将 config 替换为 `cachecodec_concat_jcb_external.json`、
`cachecodec_fusion_external.json` 或 `cachecodec_fusion_lcf_external.json`。
评测任意一组：

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/cachecodec_concat_jcb_quant_external.yaml
```

把 eval config 替换为六组表中的任意入口即可。四个短 bench 分别设置
`eval.dataset` 为 `mmlu-redux`、`ceval`、`ai2-arc` 或 `openbookqa`；建议固定
`gpu_ids: [1]`、`answer_method: generate`、`generation_config.do_sample: false`。

## 当前审计结论（2026-09）

### Sharer/Receiver 输入不是所有路径都“完全相同”

“同一个原始问题”与“完全相同的 token IDs”必须区分：

| 路径 | Sharer 输入 | Receiver 输入 | 是否相同 |
|---|---|---|---|
| Fusion，`is_do_alignment=false` | Receiver tokenizer 渲染的同一 token stream | 同一 stream | token IDs 相同；这是新版 Qwen pair 默认配置 |
| Fusion，`is_do_alignment=true` | 原 C2C `TokenAligner` 生成的 Sharer stream | `TokenAligner` 生成的 Receiver stream | 文本语义相同，IDs 不同但 mask/位置对齐 |
| Concat 训练 | 只有 `messages[:-1] + generation prompt` | prompt 加 assistant response（teacher forcing） | 不相同；Sharer 不应在训练时看到不可用的答案 |
| Concat 评测 | 同一 MCQ prompt 的 Sharer tokenizer stream | 同一 MCQ prompt 的 Receiver tokenizer stream | 原始文本相同，IDs 可不同；这是独立 virtual prefix，不做 tokenizer alignment |
| T2T | 原始 prompt，Sharer 先 decode bridge tokens | `prompt + bridge_text` 经 Receiver tokenizer 重新编码 | **不相同，这是 T2T 定义本身** |

因此不能把 T2T 或 Concat 宣称为“两侧完全相同输入”。公平性保证是：同一个原始
prompt、相同 chat-template 语义和相同 greedy/parser 评测；只有通信机制不同。

### Layer mapping 是 terminal suffix mapping

`rosetta/train/model_utils.py::last_aligned_sources(T, S, 1)` 中 `T` 是 Receiver 层数、
`S` 是 Sharer 层数：

- `S >= T`：Receiver `t` 映射 Sharer `S-T+t`，例如 `28 <- 32` 为
  `4,5,...,31`；
- `T > S`：未匹配的 Receiver 前部层复用 Sharer layer 0，随后映射到 Sharer 尾部，
  例如 `32 <- 28` 为 `0,0,0,0,0,1,...,27`；
- 每个 Receiver layer 仍只有一个 source route；这不是 tokenizer alignment，也不是
  proportional interpolation。

Fusion 的 `is_do_alignment` 默认保持 C2C-clean 的 `false`。只有 pair 确实需要跨 tokenizer
对齐时才显式设置 `true`，并选择 `first` 或 `longest`。原 C2C `Qwen3-0.6B +
Qwen2.5-0.5B` recipe 是 `false/first`；旧的 Llama pair 才使用过 `true/longest`。

### Scale 消融

固定频率表始终是低频 `1` 到高频 `8` 的二次曲线，固定 `int16 + zlib6`。评测 packet
增加两种互斥配置：

- `scale_mode: none`：直接按 `[1,8]` step 量化；
- `scale_mode: rms`：每个 layer/K 或 V packet 对 DCT 系数做一个 RMS normalization，
  将该 scalar 写入 packet metadata，再在 Receiver 端恢复。metadata 计入真实序列化
  payload bytes，因此 compression ratio 不会虚高。

scale 只在评测时打开，不参与训练，也不产生 rate loss。对应示例为
`cachecodec_concat_jcb_quant_external.yaml`（no-scale）与
`cachecodec_concat_jcb_quant_scale_external.yaml`（scale-on），Fusion 对应
`cachecodec_fusion_lcf_quant_external.yaml` 与
`cachecodec_fusion_lcf_quant_scale_external.yaml`。

### 规范化六组实验与训练数量

主矩阵名称固定为：

1. `concat`
2. `concat+jcb`
3. `concat+jcb+quant`
4. `fusion`
5. `fusion+lcf`
6. `fusion+lcf+quant`

需要 optimizer training 的是 `concat`、`concat+jcb`、`concat+lcf+qat`、`fusion`、
`fusion+lcf`、`fusion+lcf+qat` 六组；其中 QAT 训练 forward 插入 DCT+fake quant(STE)，
但不执行 zlib。主矩阵中的 `+quant` 复用对应 checkpoint，在评测期加入
DCT/int16/zlib6。这里的
`fusion+lcf` 是兼容用户实验命名的 **Sharer-side JCB transport adapter**：它恢复
Sharer geometry 后仍调用原始 C2C projector、fuser、gate 和 residual 写回，未替换
downstream fusion。代码字段仍使用 `cache_codec.type=jcb`，避免与历史 Concat-only
`LCFFirstProjector` 混淆。

训练入口配置：

```text
recipe/train_recipe/cachecodec_concat_external.json
recipe/train_recipe/cachecodec_concat_jcb_external.json
recipe/train_recipe/cachecodec_concat_lcf_qat_external.json
recipe/train_recipe/cachecodec_fusion_external.json
recipe/train_recipe/cachecodec_fusion_lcf_external.json
recipe/train_recipe/cachecodec_fusion_lcf_qat_external.json
```

评测入口配置：

```text
recipe/eval_recipe/cachecodec_concat_external.yaml
recipe/eval_recipe/cachecodec_concat_jcb_external.yaml
recipe/eval_recipe/cachecodec_concat_jcb_quant_external.yaml
recipe/eval_recipe/cachecodec_fusion_external.yaml
recipe/eval_recipe/cachecodec_fusion_lcf_external.yaml
recipe/eval_recipe/cachecodec_fusion_lcf_quant_external.yaml
```

额外基线为 `receiver_only_external.yaml`、`sharer_only_external.yaml` 和
`cachecodec_t2t_external.yaml`。QAT 的真实 zlib6 packet 只在对应
`*_qat_quant_external.yaml` 评测配置中执行。

### 八卡迁移与两卡可运行性

新服务器只需把模型、数据、运行产物放到仓库外：

```bash
export CACHECODEC_MODEL_ROOT=/server/models
export CACHECODEC_DATA_ROOT=/server/data
export CACHECODEC_RUN_ROOT=/server/cachecodec_runs
cd /server/Cachecodec2/Cachecodec
python tools/preflight_recipes.py
bash bash/train/run_cachecodec_v2_matrix_8gpu.sh
```

脚本串行训练六个需要训练的配置；保存策略是每 `100` 个 optimizer steps 保存，
删除旧 `checkpoint-*`，保留 `final/`，loss 在稳定窗口达到条件时提前停止。八卡时
每个 rank 仍加载完整 Receiver/Sharer；`CUDA_VISIBLE_DEVICES` 默认 `0..7`，可用
`GPU_IDS=0,1,...` 与 `NPROC_PER_NODE` 覆盖。

两卡启动方式：

```bash
GPU_IDS=0,1 NPROC_PER_NODE=2 MASTER_PORT=29542 \
  bash bash/train/run_cachecodec_v2_matrix_8gpu.sh
```

这在代码层面是受支持的 DDP 运行方式，但每卡 batch 不变时每 epoch 的 optimizer
step 数会随 world size 增加（Concat 约为八卡的 4 倍，Fusion 约为八卡的 4 倍）。
因此两卡可以用于 smoke test/OOM 检查，论文主训练仍应使用八卡或显式固定相同的
`max_steps`/effective batch，不能把两卡 loss 曲线直接当作八卡预算对比。当前容器没有
可用 CUDA/模型权重，所以这里只能完成静态 recipe/test 预检，不能声称两卡已经实际
训练成功；真实 GPU 服务器需先执行上述命令。

四个短 benchmark 的串行评测入口：

```bash
GPU_ID=1 bash bash/eval/run_cachecodec_v2_shortbench_matrix.sh
```

脚本顺序执行 `mmlu-redux`、`openbookqa`、`ceval`、`ai2-arc`，每个方法/数据集写入
独立目录，并固定 greedy generation。若使用本地 JSONL，可设置
`MMLU_REDUX_JSONL`、`OPENBOOKQA_JSONL`、`CEVAL_JSONL`、`AI2_ARC_JSONL`；未设置时由
evaluator 按其 Hugging Face 数据集配置加载。
不要同时把进程限制为单卡 `CUDA_VISIBLE_DEVICES=1`；若必须这样限制，需将该
recipe 临时改为 `gpu_ids: [0]`（CUDA 可见设备会重新编号）。

### 指标定义

- `accuracy`：统一 greedy 输出 parser 得到的 A/B/C/D 选项准确率。
- `payload_bytes`：实际序列化 wire payload 的字节数，包含 packet metadata；
  quant 组包含 int16 symbols、shape 和 zlib 数据。
- `compression_factor`：`original_kv_bytes / payload_bytes`，原始分母固定为
  同一 Sharer 输入、同一 dtype、同一路由的完整 K/V。
- `end_to_end_latency_ms`：Sharer prefill、JCB/DCT/量化、序列化、socketpair 传输、
  解包重建、Receiver prefix/fusion 和答案生成的总时间。

六组结果可分别回答：Concat 与 Fusion 的原始 utility 差异、JCB bottleneck
带来的压缩收益，以及固定 `[1,8]` DCT/int16/zlib6 对 accuracy、ratio 和端到端
时间的额外影响。

## QAT 训练组

QAT 是正式训练实验，不修改冻结的 Receiver/Sharer LLM，也不改变原始 C2C
projector、fuser、gate 或 residual 写回。训练时只执行 DCT、固定 `[1,8]` fake
quant 和 STE；zlib6/序列化永远不进入训练图。两组 canonical recipe 为：

| 训练组 | recipe | latent 处理 | 评测 raw | 评测 DCT/int16/zlib |
|---|---|---|---|---|
| `concat+lcf+qat` | `recipe/train_recipe/cachecodec_concat_lcf_qat_external.json` | JCB/LCF latent → sequence-DCT → fake quant(STE) → IDCT → Receiver decoder | `recipe/eval_recipe/cachecodec_concat_lcf_qat_external.yaml` | `recipe/eval_recipe/cachecodec_concat_lcf_qat_quant_external.yaml` |
| `fusion+lcf+qat` | `recipe/train_recipe/cachecodec_fusion_lcf_qat_external.json` | JCBC2C latent → sequence-DCT → fake quant(STE) → IDCT → Sharer decoder | `recipe/eval_recipe/cachecodec_fusion_lcf_qat_external.yaml` | `recipe/eval_recipe/cachecodec_fusion_lcf_qat_quant_external.yaml` |

QAT 的固定表与 PTQ 完全一致：`beta=1`、低频步长 `1`、高频步长 `8`，按
`1+7*(f/(S-1))²` 变化。`DCTFakeQuantizer` 位于
`rosetta/cachejpeg/fake_quant.py`，round 使用 straight-through estimator；训练日志
中记录 `dct_qat_quantization_mse`、`dct_qat_sequence_length` 等诊断字段。QAT 训练
不会产生 wire `payload_bytes`，也不会调用 zlib；最终 ratio 和传输时间必须使用对应的
`*_qat_quant_external.yaml` 评测配置测量。

### QAT/历史扩展训练矩阵

每组都用 OpenHermes-2.5 train split 的 500k 样本（`train_ratio=0.99`），一 epoch，
Receiver/Sharer 冻结，训练 response-token CE。每 100 个 optimizer step 保存一次；
`keep_only_latest_checkpoint=true` 删除旧的 `checkpoint-*`，最终保留 `final/`。启用
的 early stopping 使用 20-step 窗口、连续 3 个稳定窗口、`min_delta=0.001`。

| 组 | 训练 recipe |
|---|---|
| Concat raw | `cachecodec_concat_external.json` |
| Concat JCB | `cachecodec_concat_jcb_external.json` |
| Concat LCF QAT | `cachecodec_concat_lcf_qat_external.json` |
| Fusion raw | `cachecodec_fusion_external.json` |
| Fusion JCB/LCF | `cachecodec_fusion_lcf_external.json` |
| Fusion LCF QAT | `cachecodec_fusion_lcf_qat_external.json` |

### 建议的完整评测矩阵

以下每个 CacheCodec 组都应在四个短 benchmark 上运行：
`mmlu-redux`、`openbookqa`、`ceval`、`ai2-arc`。只替换 eval recipe 的
`eval.dataset` 和相应 `local_jsonl_file`，其余 prompt、greedy 生成、parser、GPU
`[1]` 保持一致。

| 评测组 | 基础 recipe | 传输边界 |
|---|---|---|
| Receiver-only | `receiver_only_external.yaml` | 无 Sharer |
| Sharer-only | `sharer_only_external.yaml` | 仅 Sharer 直接回答 |
| T2T | `cachecodec_t2t_external.yaml` | Sharer 文本 decode → Receiver tokenizer re-encode |
| Concat raw | `cachecodec_concat_external.yaml` | 完整 routed Sharer K/V |
| Concat JCB | `cachecodec_concat_jcb_external.yaml` | JCB latent 浮点 |
| Concat JCB + PTQ | `cachecodec_concat_jcb_quant_external.yaml` | JCB + DCT/int16/zlib6 |
| Concat LCF QAT | `cachecodec_concat_lcf_qat_external.yaml` | QAT checkpoint，latent 浮点 |
| Concat LCF QAT + PTQ | `cachecodec_concat_lcf_qat_quant_external.yaml` | QAT checkpoint + DCT/int16/zlib6 |
| Fusion raw | `cachecodec_fusion_external.yaml` | 完整 routed Sharer K/V |
| Fusion LCF | `cachecodec_fusion_lcf_external.yaml` | Sharer-side JCB/LCF latent 浮点 |
| Fusion LCF + PTQ | `cachecodec_fusion_lcf_quant_external.yaml` | JCB/LCF + DCT/int16/zlib6 |
| Fusion LCF QAT | `cachecodec_fusion_lcf_qat_external.yaml` | QAT checkpoint，latent 浮点 |
| Fusion LCF QAT + PTQ | `cachecodec_fusion_lcf_qat_quant_external.yaml` | QAT checkpoint + DCT/int16/zlib6 |

## 训练 loss 与 checkpoint 审计

- Concat 的 sharer 只处理 prompt；Receiver labels 中 prompt 为 `-100`，只有
  assistant response 参与 causal CE。
- Fusion 的新版 recipe 设置 `include_response=true`，最后 response section 在
  已融合 cache 上计算同一 response-token CE；不额外加入 rate、重建或 gate 正则项。
- `evaluate_model` 现在按每个 batch 的有效 label token 数加权，返回全局
  `sum(loss * valid_tokens) / sum(valid_tokens)`，避免不同 response 长度造成 batch
  平均偏差。训练日志的 `total_loss` 在无 rate loss 时等于 task CE。
- `save_steps`/`eval_steps` 作用于 optimizer step，不是 micro-batch；梯度累积完成后
  才递增 global step。early stopping 只在 optimizer step 判断，并在退出前照常写出
  `final/`。

## 延迟、TTFT 与字节统计

Cache-level wrapper 每样本记录：`sharer_prefill_ms`、`sender_encode_ms`（JCB/DCT
等编码）、`serialize_ms`、`transmit_ms`、`deserialize_ms`、`codec_decode_ms`
（packet/IDCT/JCB decoder）、`fusion_ms` 或 concat decoder、`receiver_prefill_ms`、
`receiver_first_token_ms`、`ttft_ms` 和外层完整 `end_to_end_latency_ms`。CUDA 阶段前后
调用 `torch.cuda.synchronize`，socketpair 的 CPU 序列化与传输时间也计入。
其中 `codec_decode_ms` 只计 packet/IDCT/JCB 重建，`receiver_decode_ms` 只计答案 token
decode；两者不重复计入 TTFT。

定义：

```text
TTFT = Sharer prefill
     + JCB/DCT/quant encode
     + serialize + transport + deserialize
     + IDCT/JCB decode + concat/fusion
     + Receiver prefill
     + first answer-token selection/decode
```

`end_to_end_latency_ms` 是 evaluator 外层 wall-clock，包含上述阶段和剩余调度开销；
TTFT 是可解释的阶段和，不能用它替代 E2E。`compression_factor` 始终用同一路由、
同一 dtype 的 raw Sharer K/V 字节除以实际 `len(serialize_payload(packet))`，不能用
理论 int16 大小代替 zlib 后的 wire bytes。

T2T 不产生 `payload_bytes` 或 compression ratio。其 `last_stats` 单独记录：
`sharer_decode_tokens`、`bridge_text_length`、`receiver_reencoded_tokens`、
`sharer_decode_ms`、`receiver_encode_ms`、`receiver_prefill_ms`、
`receiver_first_token_ms`、`ttft_ms` 和 `internal_end_to_end_latency_ms`。外层 evaluator
仍记录统一的 `end_to_end_latency_ms`。T2T 与 cache-level 使用同一 MCQ 原始
prompt、同一 Receiver 作为最终答题模型、同一 greedy 设置；Sharer bridge 的 token
数通过 `communication_max_new_tokens` 固定并写入结果，避免把桥接文本误当作
cache payload。

T2T 评测命令：

```bash
export CACHECODEC_MODEL_ROOT=/path/to/models
export CACHECODEC_DATA_ROOT=/path/to/data
export CACHECODEC_RUN_ROOT=/path/to/runs
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/cachecodec_t2t_external.yaml
```

该命令不读取任何 checkpoint，因而不会把 T2T 混入训练矩阵；将 recipe 中
`eval.dataset` 改为四个短 benchmark 即可得到对应结果。
