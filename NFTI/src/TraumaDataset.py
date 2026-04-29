from src.Header import Header
from src.TraumaRecord import TraumaRecord
import os
import json
import csv
import pandas as pd

class TraumaDataset:
    def __init__(self):
        """
        Initialize the TraumaDataset to hold multiple records and headers.
        """
        self.headers = []
        self.records = []
        self.custom_features = []

    def add_header(self, name, ntds_page='', definition='', timing='', data_type='', usage='', one_hot_grouping='', y=''):
        """
        Add a new Header to the dataset. Only headers with a valid data_type should be added.
        """
        if data_type:
            self.headers.append(Header(name, ntds_page, definition, timing, data_type, usage, one_hot_grouping, y))

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
                print(f"Error adding feature '{feature['header']}': Missing dependencies {missing_dependencies}")
            else:
                # All dependencies are present, proceed to add the custom feature
                self.add_header(
                    feature['header'],
                    ntds_page="",
                    definition=f"Custom feature: {feature['calculation']}",
                    timing=feature['timing'],
                    data_type=feature['data_type'],
                    usage="1",  # Marking as "1" since they are used as input
                    one_hot_grouping="",
                    y=""
                )

    def add_record(self, data_row):
        """
        Add a new TraumaRecord to the dataset. Only populate fields for headers that have a valid data_type.
        """
        # Filter valid headers
        valid_headers = [header for header in self.headers if (header.data_type and header.usage == "1")]
        prediction_headers = [header for header in self.headers if (header.data_type and header.y == "1")]

        # Create filtered records for X (input) and Y (prediction) data
        filtered_record = {header.name: data_row.get(header.name, 0) for header in valid_headers}
        prediction_record = {header.name: data_row.get(header.name, 0) for header in prediction_headers}
        
        # Add filtered record to the dataset
        self.records.append(TraumaRecord(filtered_record, prediction_record, self.custom_features))

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
    def review_and_adjust_for_biu(self):
        """
        After imputation, review each field in the dataset. 
        If its related BIU field is set, the main field should be set to 0, and the BIU field should remain unchanged.
        """
        # Loop through each record in the dataset
        for record in self.records:
            for header in self.get_headers():
                if 'BIU' in header.name:
                    # Extract the name of the sister field by removing "_BIU"
                    sister_field_name = header.name.replace('_BIU', '')
                    
                    # Check if the BIU field is set (not nan or 0) and the sister field is present
                    if header.name in record.data and sister_field_name in record.data:
                        if record.data[header.name] == 1:
                            # Set the sister field to 0 if its BIU field "NA" is set
                            record.data[sister_field_name] = 0

        print("BIU-related fields have been reviewed and adjusted.")
    
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
