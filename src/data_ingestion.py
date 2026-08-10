'''
Note -Here only read the raw data and store it in the project's data pipeline.

1. Read the raw dataset (IBM.csv)
2. Verify that the file exists
3. Load it into a Pandas DataFrame
4. Save a copy to data/raw/ (if needed)
5. Return the DataFrame to the next pipeline stage
6. Log the process
'''


#To handles file paths && load the data 

from pathlib import Path
import pandas as pd
import yaml
from src.logger import logger
import sys
from src.exception import CustomException
#data_path = "data/raw/IBM.csv"

from src.data_validation import DataValidation
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering


class DataIngestion:
    """
    Responsible for reading the raw dataset.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            
        self.file_path = Path(config["data_ingestion"]["raw_data_path"])

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset into a Pandas DataFrame.
        """
        
        try:
            logger.info("Loading dataset...")

            if not self.file_path.exists():
               raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )
            
            df = pd.read_csv(self.file_path, parse_dates=["Date"])

            logger.info("Dataset loaded successfully")


            return df
            
        except Exception as e:
            
            logger.error(e)
            raise CustomException(e, sys)
            
     


if __name__ == "__main__":

    ingestion = DataIngestion()

    df = ingestion.load_data()

    print(df.head())