import random

from src.data.safe_arithmetic_eval import safe_eval_arithmetic

class TraumaRecord:
    def __init__(self, data_row, y, custom_features):
        """
        Initializes a TraumaRecord object with a row of data and its associated headers.
        """
        self.data = data_row
        self.y = y
        self.for_testing = random.random() < 0.15  # 5% chance

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
        Calculate custom features based on the provided custom feature definitions.
        """
        for feature in custom_features:
            values = {dep: self.data.get(dep, 0) for dep in feature['dependencies']}
            
            try:
                # Custom feature formulas come from `Data/customs.csv`.
                # We evaluate them using a safe arithmetic-only evaluator.
                self.data[feature['header']] = safe_eval_arithmetic(feature['calculation'], values)
            except ZeroDivisionError:
                # print(f"Division by zero encountered in feature {feature['header']}. Setting value to NaN.")
                self.data[feature['header']] = float('nan')  # Assign NaN if division by zero occurs
            except Exception as e:
                print(f"Error calculating feature {feature['header']}: {e}")
                self.data[feature['header']] = 0  # Assign default value on error
