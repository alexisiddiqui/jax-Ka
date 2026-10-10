from pkabench.training_pools import allocate, choose_validation, min_fractions, size_bucket, subset


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


def test_allocate_is_proportional_and_exact():
    quota = allocate({'a': 700, 'b': 200, 'c': 100}, 160)
    assert sum(quota.values()) == 160 and quota == {'a': 112, 'b': 32, 'c': 16}
    assert sum(allocate({'a': 3, 'b': 3, 'c': 3}, 4).values()) == 4


def test_choose_validation_one_per_group_stratified():
    rows = [dict(id=f'g{g}-{i}', group=f'g{g}', stratum='x' if g % 3 else 'y', n_res=100 if g % 2 else 900, labelled_sites=i)
            for g in range(300) for i in range(3)]
    chosen, cells = choose_validation(rows, 30, eligible=lambda r: r['labelled_sites'] > 0)
    assert len(chosen) == 30 and len({r['group'] for r in chosen}) == 30 and all(r['labelled_sites'] > 0 for r in chosen)
    assert sorted(c['validation'] for c in cells.values()) == [5, 5, 10, 10]
    assert choose_validation(rows, 30)[0] == choose_validation(rows, 30)[0]
    assert size_bucket(128) == '128' and size_bucket(129) == '256' and size_bucket(1600) == 'over'
