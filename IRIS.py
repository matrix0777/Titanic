# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the Iris dataset from a CSV file
iris_df = pd.read_csv('C:/Users/mandl/Downloads/IRIS.csv')

# Display the first few rows of the dataset to verify the columns
print(iris_df.head())

# Scatter plot for sepal length vs sepal width
plt.figure(figsize=(10, 6))
plt.scatter(iris_df[iris_df['species'] == 'setosa']['sepal_length'], iris_df[iris_df['species'] == 'setosa']['sepal_width'], label='Setosa', marker='o')
plt.scatter(iris_df[iris_df['species'] == 'versicolor']['sepal_length'], iris_df[iris_df['species'] == 'versicolor']['sepal_width'], label='Versicolor', marker='s')
plt.scatter(iris_df[iris_df['species'] == 'virginica']['sepal_length'], iris_df[iris_df['species'] == 'virginica']['sepal_width'], label='Virginica', marker='^')

plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()

# Separate features (X) and target variable (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Convert the categorical species column into numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Convert predictions back to original species labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', classification_rep)
