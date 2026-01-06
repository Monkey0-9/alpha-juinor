def test_get_alpha_families_importable():
    from alpha_families import get_alpha_families
    alphas = get_alpha_families()
    assert isinstance(alphas, list)
    assert len(alphas) > 0
