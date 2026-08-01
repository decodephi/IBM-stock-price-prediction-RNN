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

#create a class 

from pathlib import Path
import pandas as pd

data_path = "data/raw/IBM.csv"


class DataIngestion:
    """
    Responsible for reading the raw dataset.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset into a Pandas DataFrame.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path, parse_dates=["Date"])

        print("Dataset Loaded Successfully")
        print(f"Shape : {df.shape}")

        return df


if __name__ == "__main__":

    ingestion = DataIngestion(data_path)

    df = ingestion.load_data()

    print(df.head())