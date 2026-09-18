# Cachecodec Concat 方案设计

## 目标

构建一个名为 `Cachecodec` 的纯代码项目，用于异构模型的 KV 通信：采用 concat prefix、OpenHermes 500k 训练，以及 MMLU-Redux、OpenBookQA、ARC-Challenge、C-Eval 四项短基准评测。

## 已确定的决策

- 通信架构为 cache concat，不采用 fuser 侧 KV fusion。
- `LCFProjectedKVProjector` 是唯一的 concat projector。它保留可学习的 shared encoder、K/V projections 与相互独立的 K/V decoders；不得将其改为对 latent 直接使用 `chunk` 切分。
- source 与 receiver 分别使用各自 tokenizer 独立渲染和分词 prompt。活跃 concat 路径不使用 `TokenAligner`、等长 token 序列或 token 位置一一对应关系。
- source prefill 只包含 prompt/history；receiver prefill 包含由自身 tokenizer 独立生成的 prompt/history 与带监督的 response tokens。
- receiver prefix 的位置由接收 cache 的实际长度决定。receiver 的 `position_ids` 和 `cache_position` 均从该实际 prefix 长度起算，attention mask 覆盖 prefix 与 receiver tokens。
- 模型权重、HF cache、数据集与生成的 checkpoint 均位于仓库外部。recipe 仅保存外部绝对路径或用户提供的覆盖路径。
- 实验使用已有的 C2C Python 环境运行。

## 活跃数据流

```text
OpenHermes 对话
  -> source tokenizer 渲染历史 prompt
  -> 冻结 source prefill 并捕获 pre-RoPE K
  -> LCFProjectedKVProjector encode
  -> 可学习的 K/V transport views
  -> CacheJPEG/transport decode
  -> LCFProjectedKVProjector decode_transport
  -> receiver 在紧凑 prefix 位置施加 RoPE
  -> receiver tokenizer 独立渲染完整 SFT 序列
  -> receiver CE loss 仅更新 projector
```

训练使用 OpenHermes 500k。评测使用四项固定短基准，source 与 receiver 必须独立对 prompt 分词。LongBench 和仅监督答案字母的 MCQ 训练不在本阶段范围内。

## 最小化代码边界

- 新增 concat 专用的双 prompt dataset 与 collator，不改变遗留 `ChatDataset` 或 `AlignedChatDataset` 的行为。
- 仅修改 concat 专用模型与评测路径，使其接收显式的 receiver/source tensors；从活跃路径移除 `TokenAligner` import 与 `is_do_alignment` 开关。
- 保留 `LCFProjectedKVProjector`、pre-RoPE 与 compact-RoPE 辅助逻辑。
- 仅在确认保留的活跃路径未 import 后，删除 LongBench 代码、recipe、shell launcher、日志、测试以及 fusion 专用代码。
- 保留聚焦 concat/projected-KV 的测试；将依赖对齐的测试替换为独立 prompt 契约测试。

## 验收标准

1. 仓库名称为 `Cachecodec`，且不包含模型权重或大型缓存。
2. 活跃 OpenHermes concat recipe 使用 500,000 个样本、`cache_alignment: concat` 和 `concat_projector.type: lcf_projected_kv`。
3. 活跃 concat 训练与评测路径不依赖 `TokenAligner`。
4. 活跃 shortbench recipes 仅覆盖 MMLU-Redux、OpenBookQA、ARC-Challenge、C-Eval。
5. 单元测试证明 source/receiver prompt 可有不同长度、receiver prefix position 正确，且 K/V transport views 是可学习投影。
