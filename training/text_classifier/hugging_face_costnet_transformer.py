import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load CSV file
df = pd.read_csv(
    r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\data\\classifier\\cleaned_training_data.csv",
    names=["text", "label"],
)

# Map labels to numeric values
label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label"] = df["label"].map(label_mapping)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test

model_name = "distilbert-base-uncased"
# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["text"])  # Remove original text column
dataset.set_format("torch")

# Load pre-trained model with classification head
num_labels = len(label_mapping)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="training/text_classifier/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="training/text_classifier/logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)


# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Trainer
trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["test"], compute_metrics=compute_metrics
)

# Train model
trainer.train()

results = trainer.evaluate()
print(results)

# Get predictions on test set
predictions = trainer.predict(dataset["test"])

# Extract logits and convert to class predictions
logits = predictions.predictions
pred_labels = np.argmax(logits, axis=-1)
true_labels = dataset["test"]["label"]

# Compute Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

model.save_pretrained(rf"training/text_classifier/{model_name}-model-all-classifier")
tokenizer.save_pretrained(rf"training/text_classifier/{model_name}-model-all-classifier-tokenizer")

# # Load model for inference
# classifier = pipeline("text-classification", model=f"{model_name}-model-classifier", tokenizer=f"{model_name}-model-classifier-tokenizer")

# # Test prediction
# print(classifier("This is a tax-related document"))
