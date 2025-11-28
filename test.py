import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split

# Load the trained model
model = joblib.load("model.pkl")

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["category", "rating", "text_"]]
y_train = train_df["label"]

X_test = test_df[["category", "rating", "text_"]]
y_test = test_df["label"]

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(accuracy)

cm = confusion_matrix(y_test, predictions)
labels = sorted(y_test.unique())

print("\n========== Per-Class Metrics (TP, FP, FN, TN) ==========\n")

for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    print(f"Label: {label}")
    print(f"  TP (True Positives):  {TP}")
    print(f"  FP (False Positives): {FP}")
    print(f"  FN (False Negatives): {FN}")
    print(f"  TN (True Negatives):  {TN}")
    print()