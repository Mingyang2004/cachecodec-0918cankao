# CacheCodec timing edit set

This directory is an isolated edit set.  It never modifies the source project.

Apply files in this order:

1. `rosetta/cachejpeg/transport.py`
2. `rosetta/cachejpeg/packet.py`
3. `rosetta/cachejpeg_rosetta/wrapper.py`
4. `rosetta/cachejpeg_rosetta/layer_streaming.py`
5. `rosetta/cachejpeg_rosetta/concat_layer_streaming.py`
6. `rosetta/baseline/t2t.py`
7. `script/evaluation/unified_evaluator.py`
8. add `rosetta/utils/timing.py` (new file), then merge `CONFIG_CHANGES.yaml`
   into each benchmark config.

These files are based on the Cachecodec checkout that was available while this
edit set was prepared.  Before copying to the other server, compare its source
with these files and preserve any server-specific model/projector code.  The
edits are intentionally concentrated in transport boundaries, timing fields,
and return-path bookkeeping.

The final evaluator must treat its outer synchronized `model.generate()` wall
clock as authoritative `model_e2e_ms`.  Internal timings are stage diagnostics
and must not be summed for streaming runs.

For the configured 20,000,000 bytes/s simulated link, report
`link_model_seconds` from transport stats.  It is based on serialized frame
bytes and includes the configured fixed one-way latency.

`packet.py` currently implements the fixed JCB `DCT -> int16 -> zlib6` packet
path.  Do not enable layer streaming for that packet until its target-server
codec exposes compatible `encode_layer/decode_layer` functions; the existing
streaming pipeline uses the classic layer codec interface.
