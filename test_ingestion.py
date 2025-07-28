# test_ingestion.py
from mmm.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_all_data()
    print(df.head())
