# GPU/DCT optimization: changes to make only after the strict baseline is verified

The included `packet.py` profiles DCT/quantization with CUDA synchronization so
each component is measurable.  This is intentionally not the maximum-throughput
path.  Keep two evaluation modes:

1. `timing_mode: profile`: synchronized per-stage accounting, used for the paper
   breakdown and correctness checks.
2. `timing_mode: off`: asynchronous GPU stages, used for optimized end-to-end
   throughput; report only the evaluator's externally measured `model_e2e_ms`.

For the GPU codec implementation on the target server, make these changes after
you have a matching correctness baseline:

- Never call `.item()`, `.cpu()`, `.numpy()`, or `torch.cuda.synchronize()` in
  the normal encode/decode hot path merely to collect diagnostics.  Guard those
  summaries behind `timing_mode == "profile"`.
- Use one persistent CUDA stream for DCT/quantization and one pinned CPU staging
  buffer for each in-flight layer.  Record CUDA events for GPU-stage duration;
  synchronize only when emitting the profile record or before a dependent CPU
  zlib call.
- Overlap the D2H copy of layer `i` with DCT/quantization of layer `i+1`, then
  run zlib in a CPU worker.  The timeline's union/critical path, not the sum of
  all worker service times, is the streaming e2e contribution.
- zlib6 is CPU and serial for one compressed stream.  Do not claim GPU entropy
  timing.  If using independent per-layer streams, preserve layer index, tensor
  kind, shape, dtype, and codec version in every frame; decode order must not
  alter the reconstructed cache.
- The modeled link delay must always use the final framed byte count:

  $$T_{link}=T_{fixed}+\frac{N_{pickle}+8}{B}.$$

  Here $B$ is configured bandwidth in bytes/s and the extra 8 bytes are this
  project's length header.  Do not use tensor bytes, character counts, or the
  pre-zlib byte count for that quantity.
