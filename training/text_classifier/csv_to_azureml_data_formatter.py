import csv
import json
import os

# Input CSV file
# csv_file = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\classifier\\cleaned_training_data.csv"  # Update with your CSV file
csv_file = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\LOT-classifier\\cleaned_training_data.csv"

# Output directory for AzureML format
# output_dir = "C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\classifier\\azure_custom_text_classifier"
output_dir = (
    "C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\LOT-classifier\\azure_custom_text_classifier"
)
os.makedirs(output_dir, exist_ok=True)

folder_in_container = "All_07_21_2025"


# AzureML JSON file path
json_file_path = os.path.join(output_dir, "lot_classifier_07_21_2025.json")

# JSON structure template
project_json = {
    "projectFileVersion": "2022-05-01",
    "stringIndexType": "Utf16CodeUnit",
    "metadata": {
        "projectKind": "CustomSingleLabelClassification",
        "storageInputContainerName": "spendreport-lot-classifier",
        "settings": {},
        "projectName": "lot-classifier",
        "multilingual": False,
        "description": "Project-description",
        "language": "en-us",
    },
    "assets": {"projectKind": "CustomSingleLabelClassification", "classes": [], "documents": []},
}

# Read CSV and process
label_set = set()
doc_list = []

with open(csv_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
        text, label, split = row[0], row[1].upper(), row[2]  # Convert label to uppercase to match format
        if label == "CATEGORY (LABEL)":
            continue

        # Create a folder for each label
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Save each text as a .txt file
        txt_filename = f"{idx + 1}.txt"
        txt_path = os.path.join(label_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

        # Add label to classes if not added
        if label not in label_set:
            project_json["assets"]["classes"].append({"category": label})
            label_set.add(label)

        # Add document entry to JSON
        doc_list.append(
            {
                "location": f"{folder_in_container}/{label}/{txt_filename}",
                "dataset": split,
                "language": "en-us",
                "class": {"category": label},
            }
        )

# Add documents list to JSON
project_json["assets"]["documents"] = doc_list

# Save JSON file
with open(json_file_path, "w", encoding="utf-8") as json_out:
    json.dump(project_json, json_out, indent=4)

print(f"Dataset and JSON created successfully! Files saved in: {output_dir}")
