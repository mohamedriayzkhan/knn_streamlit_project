import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Drop unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Encode target column
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
# M -> 1, B -> 0

# Split features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model.pkl")

print("✅ KNN model trained and saved as model.pkl")
