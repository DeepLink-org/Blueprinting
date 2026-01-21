from blueprinting.console import print_rich_table


def test_sseqsel_fig1():
    from blueprinting.validations.cases.seqsel_fig1 import seqsel_fig1

    ret = seqsel_fig1()
    print_rich_table(ret, caption="w+opt mem & act mem")

    assert ret["w+opt mem/rtol"].max() < 0.11
    assert ret["act mem/rtol"].max() < 0.1
