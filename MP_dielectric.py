import pandas as pd
from matminer.datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load data from the Materials Project
data = load_dataset("matbench_dielectric")
df = data.reset_index(drop=True)

# Preprocess data
features = ['formation_energy_per_atom', 'band_gap', 'total_magnetic_moment', 'number_of_atoms']
X = df[features]
y = df["e_gap"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model performance
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train R^2 score: {train_r2:.3f}")
print(f"Test R^2 score: {test_r2:.3f}")
