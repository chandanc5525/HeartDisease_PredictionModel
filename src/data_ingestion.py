import pandas as pd
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

# Function to ingest data: Converts CSV data into a DataFrame
def ingest_data(data_path: str) -> pd.DataFrame:
    
    data_path = Path(data_path)

    logger.info(f"Reading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)

    logger.info(f"Data loaded successfully with shape {df.shape}")

    return df
