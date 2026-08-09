from blueprinting.console import print_rich_table


def test_seqsel_tab5():
    from blueprinting.validation.legacy.seqsel_tab5 import seqsel_tab5

    ret = seqsel_tab5()
    print_rich_table(ret, caption="iter time")

    assert ret["iter time/rtol"].max() < 0.09
