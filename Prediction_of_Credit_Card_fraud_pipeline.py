import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib

# Step 1: Load the Data
file_path = 'anacondainstall/CSV/creditcard.csv'
data = pd.read_csv(file_path)
print("Data loaded successfully.")
print("Data columns:", data.columns)  # Check column names
print(data.head())
print("Data shape:", data.shape)

# Step 2: Exploratory Data Analysis (EDA)
print("Data Summary:")
print(data.describe())
print(data.info())
print("Missing Values:", data.isnull().sum())

# Check for outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Step 3: Data Cleaning
data.fillna(data.mean(), inplace=True)
print("Missing values handled.")

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Step 4: Feature Engineering
target_column = 'y'  # Replace this with the actual column name if different
if target_column in data.columns:
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Polynomial feature generation
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_poly = poly.fit_transform(X)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Step 5: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Train/Test split completed.")

    # Step 6: Model Selection and Training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Step 7: Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.show()

    # Step 8: Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    # Step 9: Model Deployment Plan
    joblib.dump(best_model, 'anomaly_detection_model.pkl')
    print("Model saved for deployment.")
else:
    print(f"Column '{target_column}' not found. Available columns are: {data.columns.tolist()}")
