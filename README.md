# gbm_org_analysis

Designed to analyze glioblastoma (GBM) cell and brain organoid data exported from Imaris.

More specifically this package extracts, performs relavent comparative statistics, and generates preliminary figures for the following data: 1) total number of GBM cells; 2) distance of each cell to the nearest surface of the organoid (distance to surface); 3) number of GBM cells whose distace to surface exceeds a defined invasion threshold; 4) brain organoid volume; 5) brain organoid surface area.

## Installation

```zsh
$ pip install gbm_org_analysis
```

## Reproducing experimental analysis with test dataset

1. Set up new virtual environement (if desired)

2. Download and unzip test dataset

    ```zsh
    $ curl -L https://miami.box.com/shared/static/uzm30l05xwjhx2v9qeie9ruc12siz00f.zip --output Gbm_TestData.zip
    $ unzip Gbm_TestData.zip
    ```

3. Navigate to data directory and install requirements

    ```zsh
    $ cd Gbm_TestData
    $ pip install -r requirements.txt
    ```

4. Run analysis

    ```zsh
    $ python3 test_analysis.py
    ```

Notes:
* Step 3: requirements.txt includes instructions for installing gbm_org_analysis package
* Step 4: test_analysis.py automatically exports processed data, statistical results and preliminary figures to three new sub directories
* Full datasets will be made available for download as they are published

Compiled code:

```zsh
curl -L https://miami.box.com/shared/static/uzm30l05xwjhx2v9qeie9ruc12siz00f.zip --output Gbm_TestData.zip
unzip Gbm_TestData.zip
cd Gbm_TestData
pip install -r requirements.txt
python3 test_analysis.py
```

