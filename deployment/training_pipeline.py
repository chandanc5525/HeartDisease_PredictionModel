from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.model_builder import train_and_save_pipeline
from src.logger import get_logger

logger = get_logger(__name__)

def run_pipeline():
    logger.info("Training pipeline started")

    df = ingest_data("data/raw/heart.csv")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    accuracy = train_and_save_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()
