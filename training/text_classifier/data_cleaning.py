import csv
import re

import chardet
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Uncomment the first time to download necessary NLP resources
nltk.download("stopwords")
nltk.download("wordnet")

# Input and output CSV files
# input_csv = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\classifier\\training_data.csv"
# output_csv = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\classifier\\cleaned_training_data.csv"
input_csv = "C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\LOT-classifier\\training_data.csv"
output_csv = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\LOT-classifier\\cleaned_training_data.csv"

# Allowed special characters
allowed_chars = set(".-/'\"")  # Modify this set as needed
replace_char = " "

# Define split ratios
split_required = True
train_size = 0.75  # 70% training data
val_size = 0  # 15% validation data
test_size = 0.25  # 15% test data


def clean_data():
    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()
    # stop_words = set(stopwords.words("english"))  # Add/remove stopwords as per requirement

    # Function to clean text
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"\$", " dollars ", text)  # Replace $ with "dollars"
        text = re.sub(r"%", " percent ", text)  # Replace % with "percent"
        text = re.sub(r"(?<!\d)\.(?!\d)", "", text)  # Remove periods that are not between digits
        text = re.sub(
            r"[^a-zA-Z0-9\s" + re.escape("".join(allowed_chars)) + "]", replace_char, text
        )  # Remove unwanted characters
        text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space

        # Tokenize, remove stopwords, and apply lemmatization
        words = text.split()
        # words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize
        return " ".join(words)

    # Detect encoding
    with open(input_csv, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]

    # Process CSV file
    with open(input_csv, "r", encoding=encoding) as infile, open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            text, label = row[0], row[1]
            cleaned_text = clean_text(text)  # Apply cleaning
            writer.writerow([cleaned_text, label])  # Write cleaned text with label


def delete_duplicates(df):
    df = df.drop_duplicates(subset=[df.columns[0]])
    print("Removed duplicates")
    return df


def split_data(df):
    # Split into training and temp (validation + test)
    train_data, temp_data = train_test_split(df, test_size=(val_size + test_size), random_state=42)

    if val_size > 0:
        # Split temp data into validation and test sets
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=42)

        # Add a new column 'split' indicating the dataset type
        train_data["Split"] = "Train"
        val_data["Split"] = "Validation"
        test_data["Split"] = "Test"

        # Concatenate back into a single DataFrame
        final_df = pd.concat([train_data, val_data, test_data])
    else:
        # Add a new column 'split' indicating the dataset type
        train_data["Split"] = "Train"
        temp_data["Split"] = "Test"

        # Concatenate back into a single DataFrame
        final_df = pd.concat([train_data, temp_data])

    print("Split data")
    return final_df


clean_data()

df = pd.read_csv(output_csv)
df = delete_duplicates(df)

if split_required:
    df = split_data(df)
df.to_csv(output_csv, index=False)

print("Completed!")
