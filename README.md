# Final project: Time-series data and application to stock markets {-}

Author: Nguyen Hoang An
StudentID: 200011

This project aims at familiarizing you with time-series data analysis and its application to stock markets. Datasets you will be working on are Nasdaq and Vietnam stock datasets.

## Project Structure {-}
├── data

│   ├── data-nasdaq

│   │   ├── csv

│   └── data-vn-20230228

|   |   ├── stock-historical-data

|   |   ├── ticker-overview.csv

## Installation {-}
To run this project, you need to install the dependencies in 'requirements.txt'. To do so, run the following command
```bash
pip install -r requirements.txt
```
## Download Data {-}
Download data folder [HERE](https://drive.google.com/drive/folders/13Ue1HVv3n8sNl8vurRR-BovUhbQV0ZkE?usp=sharing)

## Usage {-}
To run the project, run the following files:

To build model for nasdaq market, open the `200011-project-notebook-nasdaq.ipynb` file and run each cell. 
- Change path = `../DL4AI-200011-project/data/data-nasdaq/` to your local path, direct to `data/data-nasdaq/` folder

To build model for vietnam market, open the `200011-project-notebook-vn.ipynb` file and run each cell.
- Change path = `../DL4AI-200011-project/data/data-vn-20230228/` to your local path, direct to `data/data-vn-20230228/` folder

To run `stock_trading_app.py`:
- Change to your local:
    - `vn_path = '../DL4AI-200011-project/data/data-vn-20230228/'`
    - `nasdaq_path = '../DL4AI-200011-project/data/data-nasdaq/'`
- open cmd in `/DL4AI-200011-project/` , and run
    ```bash
    streamlit run stock_trading_app.py
    ```

Note: You have to build model for each market first on `.ipynb` files. After that turn off all `.ipynb`, and enjoy my trading app.

Note: If you want to see my HAD-GNN model, open appendix/GNN. You can also see my app video in appendix. 
