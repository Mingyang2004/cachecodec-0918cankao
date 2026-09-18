import torch

from rosetta.cachejpeg.raw_dct_prune import RawDCTPrunePacketCodec


def test_raw_dct_prune_zero_fills_high_frequency_coefficients():
    cache = ((torch.tensor([[[[0.9], [0.1], [0.05], [0.001]]]]),) * 2,)
    codec = RawDCTPrunePacketCodec(prune_ratio=0.75)
    packet = codec.encode(cache)
    assert packet["layers"][0]["key"]["stored_freq_len"] == 1
    restored = codec.decode(packet, cache)
    assert restored[0][0].shape == cache[0][0].shape
    assert not torch.allclose(restored[0][0], cache[0][0])


def test_raw_dct_prune_zero_percent_is_exact_identity_transport():
    cache = ((torch.randn(1, 2, 7, 4, dtype=torch.bfloat16), torch.randn(1, 2, 7, 4, dtype=torch.bfloat16)),)
    codec = RawDCTPrunePacketCodec(prune_ratio=0.0)
    restored = codec.decode(codec.encode(cache), cache)
    assert torch.equal(restored[0][0], cache[0][0])
    assert torch.equal(restored[0][1], cache[0][1])


def test_raw_dct_prune_hundred_percent_retains_no_coefficients():
    cache = ((torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)),)
    codec = RawDCTPrunePacketCodec(prune_ratio=1.0)
    packet = codec.encode(cache)
    assert packet["layers"][0]["key"]["stored_freq_len"] == 0
    restored = codec.decode(packet, cache)
    assert torch.count_nonzero(restored[0][0]) == 0
    assert torch.count_nonzero(restored[0][1]) == 0
