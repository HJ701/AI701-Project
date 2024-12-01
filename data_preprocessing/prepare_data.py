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
    "anterior crossbite",
    "anterior open bite",
    "crossbite",
    "crowding",
    "deep bite",
    "lower crowding",
    "lower midline shift",
    "midline shift",
    "missing teeth",
    "open bite",
    "overbite",
    "overjet",
    "posterior crossbite",
    "spacing",
    "upper midline shift",
    "upper spacing"
]

output_df = pd.DataFrame(columns=columns)

for idx, row in df.iterrows():
    pid = row["patient_id"]
    _set = row["set"]
    iotn = row["IOTN_Grade"]
    malocclusion = row["Malocclusion_Class"]
    radiograph_path = ast.literal_eval(row["radiograph_filenames"])
    # assert len(radiograph_path) == 1, f"More than one radiograph for patient {pid}"
    if len(radiograph_path) != 1:
        print(f"{len(radiograph_path)} radiographs for patient {pid}")
        continue
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
        ] + [row[col] for col in columns[6:]]

output_df.to_csv("Processed_Samples/consolidated_data.csv", index=False)


###
# pids = output_df["patient_id"].unique()
# tr_pids = np.random.choice(pids, size=int(0.7 * len(pids)), replace=False)
# val_pids = list(set(pids) - set(tr_pids))
# val_pids = np.random.choice(val_pids, size=int(0.5 * len(val_pids)), replace=False)
# test_pids = list(set(pids) - set(tr_pids) - set(val_pids))

# output_df["set"] = "train"
# output_df.loc[output_df["patient_id"].isin(val_pids), "set"] = "validation"
# output_df.loc[output_df["patient_id"].isin(test_pids), "set"] = "test"