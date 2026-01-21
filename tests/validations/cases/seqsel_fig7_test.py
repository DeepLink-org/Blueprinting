from blueprinting.console import print_rich_table


def test_sseqsel_fig7():
    from blueprinting.validations.cases.seqsel_fig7 import seqsel_fig7

    ret = seqsel_fig7()
    print_rich_table(ret, caption="act mem")

    assert ret["act mem/rtol"].max() < 0.31
