import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# 2. Separate input and output
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# 7. Save the trained model
joblib.dump(model, "crop_model.pkl")

print("Model saved as crop_model.pkl")