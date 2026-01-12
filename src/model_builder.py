from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from src.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_DIR = Path("artifacts/models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    logger.info("Building full ML pipeline")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("model", model)
    ])

    # Applying SMOTE AFTER preprocessing but BEFORE model fit ----
    logger.info("Applying SMOTE on training data")
    X_train_processed = pipeline[:-1].fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(
        X_train_processed, y_train
    )

    model.fit(X_train_res, y_train_res)

    # Reattach trained model
    pipeline.named_steps["model"] = model

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path = ARTIFACT_DIR / "pipeline.pkl"
    joblib.dump(pipeline, model_path)

    logger.info(f"Pipeline saved at {model_path}")
    logger.info(f"Validation accuracy: {acc}")

    return acc
