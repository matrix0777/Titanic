# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the sales dataset from a CSV file with spaces in the path
sales_data = pd.read_csv(r'C:\Users\mandl\Downloads\advertising.csv')

# Display the first few rows of the dataset to verify the columns
print(sales_data.head())

# Split the data into features (X) and target variable (y)
X = sales_data.drop('Sales', axis=1)
y = sales_data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the predictions
plt.scatter(X_test['TV'], y_test, color='black', label='Actual Sales')
plt.scatter(X_test['TV'], y_pred, color='blue', label='Predicted Sales')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.legend()
plt.show()
