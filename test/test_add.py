from gbm_org_analysis import add

def test_add():
    observed = add(3, 8)
    expected = 11
    assert observed == expected
