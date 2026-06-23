import random
import warnings

from src.data.missing_values import MISSING, field_value_from_row, is_missing
from src.data.safe_arithmetic_eval import safe_eval_arithmetic

class TraumaRecord:
    _calc_warnings_issued = set()
    def __init__(self, data_row, y, custom_features, *, assign_split=False, test_fraction=0.15):
        """
        Initializes a TraumaRecord object with a row of data and its associated headers.
        """
        self.data = data_row
        # Pre-scale working copy used for imputation and derived feature formulas.
        self.base_data = dict(data_row)
        self.y = y
        self.for_testing = random.random() < test_fraction if assign_split else False

        self.calculate_custom_features(custom_features)

    def get_value(self, key):
        """
        Get the value for a specific key (header name) in the record.
        """
        return self.data.get(key, None)

    def __repr__(self):
        """
        Return a string representation of the record for easy debugging.
        """
        return f"Data: {self.data}, Y: {self.y}, For Testing: {self.for_testing}"
    
    def calculate_custom_features(self, custom_features):
        """
        Calculate custom features from pre-scale base_data (raw physiologic values).
        """
        for feature in custom_features:
            values = {
                dep: field_value_from_row(self.base_data, dep) for dep in feature["dependencies"]
            }
            if any(is_missing(value) for value in values.values()):
                self.base_data[feature["header"]] = MISSING
                self.data[feature["header"]] = MISSING
                continue

            try:
                result = safe_eval_arithmetic(feature["calculation"], values)
                self.base_data[feature["header"]] = result
                self.data[feature["header"]] = result
            except ZeroDivisionError:
                warning_key = (feature['header'], 'ZeroDivisionError')
                if warning_key not in TraumaRecord._calc_warnings_issued:
                    warnings.warn(
                        f"Division by zero encountered in custom feature '{feature['header']}'. "
                        "Affected records will be set to NaN.",
                        UserWarning,
                        stacklevel=2,
                    )
                    TraumaRecord._calc_warnings_issued.add(warning_key)
                self.base_data[feature["header"]] = MISSING
                self.data[feature["header"]] = MISSING
            except Exception as e:
                warning_key = (feature["header"], str(e))
                if warning_key not in TraumaRecord._calc_warnings_issued:
                    warnings.warn(
                        f"Error calculating custom feature '{feature['header']}': {e}. "
                        "Affected records will be set to NaN.",
                        UserWarning,
                        stacklevel=2,
                    )
                    TraumaRecord._calc_warnings_issued.add(warning_key)
                self.base_data[feature["header"]] = MISSING
                self.data[feature["header"]] = MISSING
