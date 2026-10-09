from pkabench.training_pools import min_fractions, subset


def test_subsets_are_nested_stratified_and_deterministic():
    strata = {f'g{i}': ('a' if i % 4 else 'b') for i in range(400)}
    entry = min_fractions(strata); assert entry == min_fractions(strata)
    rows = [dict(group=g, stratum=s, min_fraction=entry[g]) for g, s in strata.items()]
    sizes = {}
    for f in (0.1, 0.5, 0.75, 1.0):
        chosen = subset(rows, f); sizes[f] = {r['group'] for r in chosen}
        assert sum(r['stratum'] == 'b' for r in chosen) == int(100 * f) and sum(r['stratum'] == 'a' for r in chosen) == int(300 * f)
    assert sizes[0.1] < sizes[0.5] < sizes[0.75] < sizes[1.0] and len(sizes[1.0]) == 400


def test_seed_changes_order():
    strata = {f'g{i}': 'a' for i in range(50)}
    assert min_fractions(strata, seed=1) != min_fractions(strata, seed=2)
