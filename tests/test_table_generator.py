# tests/test_table_generator.py
import pytest
from modules.table_generator import generate_table_data_advanced

def test_generate_table_data_advanced():
    config = [
        {
            "name": "num_col",
            "type": "numeric",
            "dist": "normal",
            "mean": 0.0,
            "std": 1.0
        },
        {
            "name": "cat_col",
            "type": "categorical",
            "categories": ["cat1", "cat2", "cat3"]
        }
    ]
    df = generate_table_data_advanced(num_samples=10, columns_config=config)
    assert len(df) == 10
    assert "num_col" in df.columns
    assert "cat_col" in df.columns
    assert all(df["cat_col"].isin(["cat1", "cat2", "cat3"]))
