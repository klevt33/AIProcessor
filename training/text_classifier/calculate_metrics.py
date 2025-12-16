import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# === File Config ===
file_path = r"C:\\Users\\VamsiMalneedi(Aspire\\M_Vamsi\\Projects\\Spend_Report\\laban-testing\\classifier_output_validated.xlsx"
metrics_sheet_name = "Classification Metrics"

# === Load the original data ===
df = pd.read_excel(file_path)

if "GT_CLASS" not in df.columns or "CLASS" not in df.columns:
    raise ValueError("The Excel file must contain 'GT_CLASS' and 'CLASS' columns.")

y_true = df["GT_CLASS"]
y_pred = df["CLASS"]
labels = sorted(set(y_true) | set(y_pred))

# === Metrics Calculation ===
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Add totals for rows and columns (sum of predicted and actual)
conf_df["TOTAL"] = conf_df.sum(axis=1)  # Row sums (Actual totals)
conf_df.loc["TOTAL"] = conf_df.sum(axis=0)  # Column sums (Predicted totals)

# === Calculate classification report and overall metrics ===
report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()

overall_df = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1 Score"],
        "Value": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        ],
    }
)

# === Load workbook and write results to a new sheet ===
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
    # Write Confusion Matrix with totals to the new sheet
    conf_df.to_excel(writer, sheet_name=metrics_sheet_name, startrow=0, index=True)

    # Write Classification Report (class-wise metrics)
    start_row_report = len(conf_df) + 3  # Leave space after the confusion matrix
    report_df.to_excel(writer, sheet_name=metrics_sheet_name, startrow=start_row_report, index=True)

    # Write Overall Metrics (Accuracy, Precision, Recall, F1 Score)
    start_row_overall = start_row_report + len(report_df) + 3
    overall_df.to_excel(writer, sheet_name=metrics_sheet_name, startrow=start_row_overall, index=False)

print(f"Metrics including confusion matrix and totals written to new sheet '{metrics_sheet_name}' in '{file_path}'")
