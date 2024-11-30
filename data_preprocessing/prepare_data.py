"""
Here we create a consolidated CSV that contains all the data from the different sources.
Paths to images, paths to radiographs, texts, and labels, etc.
"""

import ast

import pandas as pd

df = pd.read_csv("Processed_Samples/processed_data.csv")

columns = [
    "patient_id",
    "set",
    "image_path",
    "radiograph_path",
    "IOTN_grade",
    "malocclusion_class",
] + df.columns[-14:].tolist()

output_df = pd.DataFrame(columns=columns)

for idx, row in df.iterrows():
    pid = row["patient_id"]
    _set = row["set"]
    iotn = row["IOTN_Grade"]
    malocclusion = row["Malocclusion_Class"]
    radiograph_path = ast.literal_eval(row["radiograph_filenames"])
    assert len(radiograph_path) == 1, f"More than one radiograph for patient {pid}"
    radiograph_path = radiograph_path[0]
    img_paths = ast.literal_eval(row["image_filenames"])
    for img_path in img_paths:
        output_df.loc[len(output_df)] = [
            pid,
            _set,
            img_path,
            radiograph_path,
            iotn,
            malocclusion,
        ] + row[-14:].tolist()

output_df.to_csv("Processed_Samples/consolidated_data.csv", index=False)
