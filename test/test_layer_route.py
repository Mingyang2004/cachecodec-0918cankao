from rosetta.train.model_utils import last_aligned_sources


def _single_source(mapping):
    return [mapping[layer][0] for layer in sorted(mapping)]


def test_last_aligned_route_uses_terminal_suffix_when_source_is_deeper():
    assert _single_source(last_aligned_sources(3, 5)) == [2, 3, 4]


def test_last_aligned_route_maps_unmatched_receiver_front_to_source_zero():
    assert _single_source(last_aligned_sources(5, 3)) == [0, 0, 0, 1, 2]
