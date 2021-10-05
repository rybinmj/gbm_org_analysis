import os
from gbm_org_analysis import Data


def test_init():
    gbm_data_path: str = os.environ.get("GBM_DATA")
    input_path: str = os.path.join(
        gbm_data_path, "Data_Raw", "200407_gbm22_batch2_group0_org1_Statistics"
    )

    data: Data = Data(input_path)
    assert data.name == "f114", "Name is incorrect"
    assert data.org_str == "org1"
    assert data.org_num == 1
