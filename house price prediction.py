import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Sample Dataset
data = {
    'square_feet': [1500, 1600, 1700, 1800, 2000, 2100, 2500],
    'bedrooms': [3, 3, 3, 4, 4, 4, 5],
    'bathrooms': [2, 2, 2, 3, 3, 3, 4],
    'price': [300000, 320000, 340000, 360000, 400000, 420000, 500000]
}

df = pd.DataFrame(data)
print("📊 Dataset:\n", df)

# features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]  # Features
y = df['price']  # Target

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Print actual vs predicted
print("\n📈 Actual vs Predicted Prices:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: ₹{actual}, Predicted: ₹{predicted:.2f}")

# Model parameters
print("\n🧠 Model Details:")
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Ideal line
plt.grid(True)
plt.show()
