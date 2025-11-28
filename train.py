from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

def clean_data(df):
    print("Cleaning Data.")

    # Check the data types
    print("Data Types:")
    print(df.dtypes)

    # Check for missing values in each column
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Drop rows with any missing values
    df_clean = df.dropna()

    # Verify that missing rows are removed
    print("\nMissing Values After Cleaning:")
    print(df_clean.isnull().sum())
    print("\nData processing done!")
    print("====================================")
    return df_clean

def main():
    #For training the model 
    df = pd.read_csv('fake reviews dataset.csv')

    X = df[["category", "rating", "text_"]]
    y = df['label']

    df = clean_data(df)

    X_train, X_test, y_train, y_test =  train_test_split(X, y)

    # Combine X and y for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)

    # Save to CSV
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("text", TfidfVectorizer(), "text_"),
            ("num", "passthrough", ["rating"])
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("svm", SVC(kernel="linear"))
    ])
    print("====================================")
    print("Started Model Training!")
    
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl") 

    print("Model Done Training. Exported to \"model.pkl\"")

if __name__ == "__main__":
    main()