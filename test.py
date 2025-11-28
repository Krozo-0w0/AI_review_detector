import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

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