import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================================
# STEP 1: LOADING DATA & INITIAL EXPLORATION
# =====================================================================
df = pd.read_csv("employee_attrition.csv")

print("--- Initial Dataset Shape ---")
print(df.shape)

print("\n--- Data Types & Missing Values Summary ---")
print(df.info())

# =====================================================================
# STEP 2: DATA CLEANING
# =====================================================================
# Drop columns that have constant values or are unique identifiers
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])

print("\n--- Shape After Dropping Constant Columns ---")
print(df.shape)

# Convert our target text labels into binary numbers
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

print("\n--- Attrition Label Value Counts ---")
print(df["Attrition"].value_counts())

# =====================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (CORRELATIONS)
# =====================================================================
numeric_df = df.select_dtypes(include=["int64", "float64"])
correlations = numeric_df.corr()["Attrition"].sort_values()

print("\n--- Feature Correlations with Attrition ---")
print(correlations)

# =====================================================================
# STEP 4: FEATURE ENGINEERING (ONE-HOT ENCODING)
# =====================================================================
# Transform categorical text features into mathematical columns
# =====================================================================
# STEP 4: FEATURE ENGINEERING (ONE-HOT ENCODING)
# =====================================================================
# Transform categorical text features into mathematical columns (Fixed to include EducationField)
df_encoded = pd.get_dummies(df, columns=["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"], drop_first=True)

print("\n--- Shape After One-Hot Encoding Text Variables ---")
print(df_encoded.shape)


# =====================================================================
# STEP 5: DATA SPLITTING FOR MACHINE LEARNING
# =====================================================================
# Separate our predictive features (X) from our target label (y)
X = df_encoded.drop(columns=["Attrition"])
y = df_encoded["Attrition"]

# Split the dataset into 80% Training data and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Final Machine Learning Data Split Verification ---")
print("Training Data Features Shape:", X_train.shape)
print("Testing Data Features Shape:", X_test.shape)


# =====================================================================
# STEP 6: BALANCING DATA & TRAINING THE PREDICTIVE MODEL
# =====================================================================
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Initialize and train the Random Forest Classifier model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_bal, y_train_bal)

# Evaluate model performance on the untouched test data
y_pred = model.predict(X_test)
print("\n--- Model Accuracy Score ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Save the trained model and the column layout structures as artifacts for our dashboard
joblib.dump(model, 'attrition_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("\nModel artifacts saved successfully as 'attrition_model.pkl' and 'model_columns.pkl'!")
