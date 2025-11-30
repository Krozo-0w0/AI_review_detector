import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the trained model
model = joblib.load("model.pkl")

def predict_single_review():
    """
    Function to get a single review input from user and make prediction
    """
    print("Enter the details for the AI review:")
    
    # Get input from user
    category = input("Enter category: ")
    rating = float(input("Enter rating: "))
    text = input("Enter review text: ")
    
    # Create a DataFrame with the same structure as training data
    input_data = pd.DataFrame({
        "category": [category],
        "rating": [rating],
        "text_": [text]
    })
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        
        # Try to get probabilities, but handle case where it's not available
        try:
            probability = model.predict_proba(input_data)
            print(f"\n=== Prediction Result ===")
            print(f"Category: {category}")
            print(f"Rating: {rating}")
            print(f"Review Text: {text}")
            print(f"Predicted Label: {prediction[0]}")
            print(f"Prediction Probabilities: {probability[0]}")
        except AttributeError:
            # If predict_proba is not available, just show the prediction
            print(f"\n=== Prediction Result ===")
            print(f"Category: {category}")
            print(f"Rating: {rating}")
            print(f"Review Text: {text}")
            print(f"Predicted Label: {prediction[0]}")
            print("Probability scores not available for this model type")
        
        return prediction[0]
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def batch_test_mode():
    """
    Original functionality for testing on the full dataset
    """
    try:
        train_df = pd.read_csv("train_data.csv")
        test_df = pd.read_csv("test_data.csv")

        X_test = test_df[["category", "rating", "text_"]]
        y_test = test_df["label"]

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Overall Accuracy: {accuracy:.4f}")

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
            
    except Exception as e:
        print(f"Error in batch test: {e}")

def main():
    """
    Main function with menu options
    """
    print("=== AI Review Classifier ===")
    print("1. Test single review")
    print("2. Run batch test on test dataset")
    print("3. Exit")
    
    while True:
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            predict_single_review()
        elif choice == "2":
            batch_test_mode()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Run the main function
if __name__ == "__main__":
    main()