from pkatrain.parameter_sweep import GQT,PKAI,gqt_count,pkai_count


def test_registered_parameter_counts():
    for width,ff,count in GQT.values():assert gqt_count(width,ff)==count
    for hidden,count in PKAI.values():assert pkai_count(hidden)==count


def test_capacity_tiers_increase():
    assert [value[2] for value in GQT.values()]==sorted(value[2] for value in GQT.values())
    assert [value[1] for value in PKAI.values()]==sorted(value[1] for value in PKAI.values())
