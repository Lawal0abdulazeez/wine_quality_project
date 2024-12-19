# src/data_quality/data_validator.py
import pandas as pd
import great_expectations as ge
from typing import Dict, List

class DataValidator:
    def __init__(self, config):
        self.config = config
        
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate dataset using Great Expectations"""
        ge_df = ge.from_pandas(df)
        
        expectations = [
            ge_df.expect_column_values_to_not_be_null(column)
            for column in df.columns
        ]
        
        # Add specific expectations for wine quality data
        expectations.extend([
            ge_df.expect_column_values_to_be_between('quality', 0, 10),
            ge_df.expect_column_values_to_be_between('alcohol', 0, 20),
            ge_df.expect_column_values_to_be_between('pH', 2, 5)
        ])
        
        return ge_df.validate()
