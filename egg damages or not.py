#egg damages or not
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset (replace this with your actual dataset)
data = {
    'Image_Path': ["F:\DATA\not_damaged_1.jpg", 'F:\DATA\damaged_1.jpg', 'F:\DATA\not_damaged_2.jpg'],
    'Label': ['1','0','1']
}

df = pd.DataFrame(data)

# Data preprocessing
# In a real-world scenario, you would need to preprocess the images.
# For simplicity, we'll use image paths and a label indicating whether the egg is damaged or not.

# Convert labels to numerical values
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Image_Path'], df['Label'], test_size=0.2, random_state=42)

# Feature extraction (using a simple bag-of-words approach for demonstration)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

# Model evaluation
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
