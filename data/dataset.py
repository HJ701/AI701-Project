# data/dataset.py

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import logging

class OrthoAIDataset(Dataset):
    def __init__(self, csv_file, image_dir, radiograph_dir, set_type='train'):
        """
        Args:
            csv_file (str): Path to the structured_data.csv file.
            image_dir (str): Directory containing intraoral images.
            radiograph_dir (str): Directory containing radiograph images.
            set_type (str): One of 'train', 'validation', or 'test'.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['set'] == set_type].reset_index(drop=True)
        self.image_dir = image_dir
        self.radiograph_dir = radiograph_dir

        # Define subclass diagnosis columns
        self.subclass_columns = [
            'anterior crossbite',
            'anterior open bite',
            'crossbite',
            'crowding',
            'deep bite',
            'lower crowding',
            'lower midline shift',
            'midline shift',
            'missing teeth',
            'open bite',
            'posterior crossbite',
            'spacing',
            'upper midline shift'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract data for the given index
        row = self.data.loc[idx]
        intraoral_filenames = ast.literal_eval(row['image_filenames'])  # Convert string list to actual list
        radiograph_filenames = ast.literal_eval(row['radiograph_filenames'])  # Convert string list to actual list
        clinical_notes = row['clinical_notes']

        # Labels
        iotn_grade = row['IOTN_Grade']
        malocclusion_class_encoded = row['Malocclusion_Class_Encoded']

        # Subclass labels as a multi-hot vector
        subclass_labels = row[self.subclass_columns].values.astype(float)  # Ensure it's float for tensor conversion

        # Convert subclass_labels to tensor
        subclass_labels = torch.tensor(subclass_labels, dtype=torch.float32)

        # Convert IOTN_Grade and Malocclusion_Class_Encoded to tensors
        iotn_grade = torch.tensor(iotn_grade, dtype=torch.long)
        malocclusion_class_encoded = torch.tensor(malocclusion_class_encoded, dtype=torch.long)

        # Return all relevant data
        return intraoral_filenames, radiograph_filenames, clinical_notes, iotn_grade, malocclusion_class_encoded, subclass_labels