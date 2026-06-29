import random

from src.Header import Header
from src.TraumaRecord import TraumaRecord
from src.data.missing_values import MISSING, biu_indicates_missing, field_value_from_row
from src.dataset_validation import (
    warn_custom_feature_availability,
    warn_data_column_schema_gaps,
    warn_schema_data_coverage,
)
import os
import json
import csv
import warnings
import pandas as pd

class TraumaDataset:
    def __init__(self):
        """
        Initialize the TraumaDataset to hold multiple records and headers.
        """
        self.headers = []
        self.records = []
        self.custom_features = []
        self.transform_state = None
        self.imputation_state = None
        self.cohort_state = None

    def add_header(self, name, ntds_page='', definition='', timing='', data_type='', load='', usage='', y=''):
        """
        Add a new Header to the dataset. Only headers with a valid data_type should be added.
        """
        if data_type:
            self.headers.append(Header(name, ntds_page, definition, timing, data_type, load, usage, y))

    @staticmethod
    def load_custom_features(feature_file):
        """
        Load custom features from a CSV file into the dataset.
        Each custom feature has a header, timing, data_type, calculation, and space-separated dependencies.
        """
        custom_features = []
        with open(feature_file, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                custom_features.append({
                    'header': row['Header'],
                    'timing': row['Timing'],
                    'data_type': row['Type'],
                    'calculation': row['Calculation'],
                    'dependencies': row['Dependencies'].split()  # Splitting by space
                })
        return custom_features

    def add_custom_features(self, feature_file):
        self.custom_features = self.load_custom_features(feature_file)

        # Get the list of current header names
        existing_headers = [header.name for header in self.headers]

        for feature in self.custom_features:
            # Check if all dependencies exist in the current headers
            missing_dependencies = [dep for dep in feature['dependencies'] if dep not in existing_headers]

            if missing_dependencies:
                warnings.warn(
                    f"Cannot register custom feature '{feature['header']}': "
                    f"missing dependencies {missing_dependencies}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # All dependencies are present, proceed to add the custom feature
                self.add_header(
                    feature['header'],
                    ntds_page="",
                    definition=f"Custom feature: {feature['calculation']}",
                    timing=feature['timing'],
                    data_type=feature['data_type'],
                    load="1",  # Loaded into memory and used as a model input
                    usage="1",
                    y=""
                )

    def validate_build(self, header_info, data_columns):
        """
        Emit warnings for schema/data mismatches and custom features that cannot
        be calculated from the headers available in this dataset build.
        """
        warn_schema_data_coverage(header_info, data_columns)
        warn_data_column_schema_gaps(header_info, data_columns)
        if self.custom_features:
            warn_custom_feature_availability(self.custom_features, self.headers)

    def add_record(self, data_row, *, assign_split=False, test_fraction=0.15):
        """
        Add a new TraumaRecord to the dataset. Only populate fields for headers that have a valid data_type.
        """
        # Filter valid headers. ``load`` decides what is populated into memory
        # (record.data); ``usage`` is reserved for downstream training selection.
        valid_headers = [header for header in self.headers if (header.data_type and header.load == "1")]
        prediction_headers = [header for header in self.headers if (header.data_type and header.y == "1")]

        # Missing columns or absent values use NaN; valid zeros are preserved.
        filtered_record = {
            header.name: field_value_from_row(data_row, header.name)
            for header in valid_headers
        }
        prediction_record = {
            header.name: field_value_from_row(data_row, header.name)
            for header in prediction_headers
        }
        
        # Add filtered record to the dataset
        self.records.append(
            TraumaRecord(
                filtered_record,
                prediction_record,
                self.custom_features,
                assign_split=assign_split,
                test_fraction=test_fraction,
            )
        )

    def assign_train_test_split(self, test_fraction=0.15, random_state=42):
        """Assign the holdout flag after cohort filtering."""
        rng = random.Random(random_state)
        for record in self.records:
            record.for_testing = rng.random() < test_fraction

    def get_headers(self):
        """
        Return all headers.
        """
        return self.headers

    def get_records(self):
        """
        Return all records.
        """
        return self.records

    def __repr__(self):
        """
        Represent the entire dataset for debugging.
        """
        return f"TraumaDataset with {len(self.records)} records and {len(self.headers)} headers"
    def recalculate_custom_features(self):
        """Recompute all custom feature columns for every record from pre-scale base_data."""
        if getattr(self, "transform_state", None) and self.transform_state.get("applied"):
            warnings.warn(
                "Skipping custom feature recalculation because dataset transforms were already "
                "applied. Reload a pre-transform pickle to re-engineer features.",
                UserWarning,
                stacklevel=2,
            )
            return
        if not self.custom_features:
            return
        for record in self.records:
            if not hasattr(record, "base_data") or record.base_data is None:
                record.base_data = dict(record.data)
            record.calculate_custom_features(self.custom_features)

    def review_and_adjust_for_biu(self):
        """
        When a BIU field indicates the sister value was not observed,
        set the sister field to NaN (not 0).
        """
        for record in self.records:
            for header in self.get_headers():
                if "BIU" not in header.name:
                    continue

                sister_field_name = header.name.replace("_BIU", "")
                if header.name not in record.data or sister_field_name not in record.data:
                    continue

                if biu_indicates_missing(record.data[header.name]):
                    record.data[sister_field_name] = MISSING
                    if hasattr(record, "base_data") and record.base_data is not None:
                        record.base_data[sister_field_name] = MISSING

        print("BIU-related fields have been reviewed and adjusted.")

    def finalize_after_imputation(self):
        """Apply BIU rules and refresh derived custom features after imputation."""
        self.review_and_adjust_for_biu()
        self.recalculate_custom_features()
    
def generate_json_from_headers(headers):
    """
    Generate a JSON template for user input with all headers initialized to 0.
    """
    os.makedirs('Testing', exist_ok=True)

    filtered_headers = [header.name for header in headers if header.usage == '1']
    
    headers_dict = {header: 0 for header in filtered_headers}

    json_file_path = os.path.join('Testing', 'user_input_template.json')

    with open(json_file_path, 'w') as json_file:
        json.dump(headers_dict, json_file, indent=4)
    
    print(f"JSON file with headers initialized to 0 saved at: {json_file_path}")

def load_from_json(json_file_path):
    """
    Load a TraumaDataset object from a JSON file with user input.
    """
    with open(json_file_path, 'r') as json_file:
        user_input_data = json.load(json_file)

    trauma_dataset = TraumaDataset()

    for header, value in user_input_data.items():
        trauma_dataset.add_header(header)
    
    trauma_dataset.add_record(user_input_data)  # Convert JSON to a TraumaRecord

    return trauma_dataset
