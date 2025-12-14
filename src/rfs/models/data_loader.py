import pandas as pd
import logging
from src.rfs.db.connector import get_db_engine

logger = logging.getLogger("model.data_loader")


def load_training_data() -> pd.DataFrame:
    """
    Features tablosundan (Gold Layer) eğitim verisini çeker.
    """
    engine = get_db_engine()
    try:
        # Sadece features şemasından çekiyoruz
        query = "SELECT * FROM features.laptops_final"
        df = pd.read_sql(query, engine)
        logger.info(f"Training data loaded: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise
