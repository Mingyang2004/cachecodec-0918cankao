# Projected-KV Allocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 `test/test_lcf_projected_kv.py` 为主契约，为 concat + `LCFProjectedKVProjector` 增加专用的 layer×K/V alpha allocator，同时保持现有 projector、legacy quantizer 和 checkpoint API 可用。

**Architecture:** 保留现有 `AdaptiveCoefficientQuantizer` 作为旧路径兼容实现，新增 `ProjectedKVAdaptiveQuantizer` 专门处理单 pseudo-head 的 projected transport K/V。新 allocator 使用每频带四类统计、layer/KV identity、local/global context 和 hard Gumbel-Softmax；训练入口只在 concat + `lcf_projected_kv` 时选择该实现。

**Tech Stack:** Python 3.10、PyTorch、Transformers、pytest、C2C conda 环境。

---

### Task 1: 固化 Projected-KV allocator 契约

**Files:**
- Modify: `test/test_lcf_projected_kv.py`
- Test: `test/test_lcf_projected_kv.py`

- [ ] **Step 1: 写入失败测试**

新增测试，要求 `ProjectedKVAdaptiveQuantizer`：

```python
quantizer = ProjectedKVAdaptiveQuantizer(
    num_layers=2,
    config=resolve_adaptive_quant_table_config(
        {"enabled": True, "allocator_type": "projected_layer_kv", "feature_bands": 8}
    ),
)
result = quantizer(projected_transport_cache)
assert result.alpha.shape == (1, 2, 2, 1)
assert quantizer.num_groups == 4
assert quantizer.local_encoder[0].in_features == 52
```

同时要求不同 layer/KV group 获得独立 logits、评测使用确定性 argmax、任务和 rate 梯度到达 shared encoder、K/V projections、allocator 与 entropy model。

- [ ] **Step 2: 运行测试确认 RED**

Run: `conda run -n c2c python -m pytest -q test/test_lcf_projected_kv.py`

Expected: 因 `ProjectedKVAdaptiveQuantizer` 尚不存在而在 collection 或目标测试处失败。

### Task 2: 实现专用 layer×K/V allocator

**Files:**
- Modify: `rosetta/model/adaptive_quant_table.py`
- Test: `test/test_lcf_projected_kv.py`

- [ ] **Step 1: 实现频谱特征与 local/global 网络**

实现以下固定结构：

```text
normalized DCT coefficients
  -> 8 bands × [mean_abs, rms, soft_near_zero, mean_log1p_abs] = 32
  -> layer embedding 16 + KV embedding 4 = 20
  -> local input 52 -> 64 -> 128
  -> mean(local over groups) -> global 128 -> 128 -> 128
  -> concat local/global 256 -> 128 -> 5 alpha logits
```

- [ ] **Step 2: 保持量化结果 API 兼容**

返回现有 `AdaptiveQuantTableResult`，其中：

```text
past_key_values: tuple[(K,V)]
alpha/table_indices/scale: [B,L,2,1]
rounded_symbols: [B,L,2,1,S,C]
```

- [ ] **Step 3: 运行目标测试确认 GREEN**

Run: `conda run -n c2c python -m pytest -q test/test_lcf_projected_kv.py`

Expected: 全部通过。

### Task 3: 训练入口选择正确 allocator

**Files:**
- Modify: `script/train/SFT_train.py`
- Modify: `recipe/train_recipe/C2C_openhermes_50k_concat_lcf_projected_kv_adaptive_quant.json`
- Test: `test/test_lcf_projected_kv.py`

- [ ] **Step 1: 写入失败测试**

测试工厂函数在 `cache_alignment=concat` 且 `concat_projector.type=lcf_projected_kv` 时返回 `ProjectedKVAdaptiveQuantizer`；其他旧配置继续返回 `AdaptiveCoefficientQuantizer`。

- [ ] **Step 2: 运行测试确认 RED**

Run: `conda run -n c2c python -m pytest -q test/test_lcf_projected_kv.py -k factory`

Expected: 因工厂函数不存在或仍返回 legacy quantizer 而失败。

- [ ] **Step 3: 实现最小工厂并更新 recipe**

新增单一工厂入口，并在主 recipe 中写入：

```json
"allocator_type": "projected_layer_kv"
```

保留旧配置默认值，避免破坏遗留 checkpoint 与测试。

- [ ] **Step 4: 运行目标测试确认 GREEN**

Run: `conda run -n c2c python -m pytest -q test/test_lcf_projected_kv.py`

Expected: 全部通过。

### Task 4: 兼容性验证

**Files:**
- Test: `test/test_lcf_projected_kv.py`
- Test: `test/test_adaptive_quant_table.py`
- Test: `test/test_concat_cache_alignment.py`

- [ ] **Step 1: 运行聚焦测试**

Run: `conda run -n c2c python -m pytest -q test/test_lcf_projected_kv.py test/test_adaptive_quant_table.py test/test_concat_cache_alignment.py`

Expected: 全部通过，无 collection error。

- [ ] **Step 2: 运行编译检查**

Run: `conda run -n c2c python -m compileall -q rosetta script/train/SFT_train.py`

Expected: exit code 0。

- [ ] **Step 3: 审核主链约束**

确认 `lcf_projected_kv` 仍调用 `key_projection`、`value_projection` 和 `decode_transport`，并且新 allocator 的 group 数严格等于 `2 × receiver_layers`。
