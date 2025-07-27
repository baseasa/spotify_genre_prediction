import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


df = pd.read_csv(r"C:\spyder\datasets\spotify+top50\top50.csv", encoding='ISO-8859-1')
print(df.head())

print("shape:", df.shape)

print("\nInfo")
print(df.info())

print("\nDescribe")
print(df.describe(include = 'all'))

print("Columns:", df.columns.tolist())

print("Genres:")
print(df['Genre'].value_counts())

selected_features = [
    'Beats.Per.Minute',
    'Energy',
    'Danceability',
    'Loudness..dB..',
    'Valence.',
    'Acousticness..',
    'Speechiness.',
    'Popularity'
    ]

X = df[selected_features]
y = df['Genre']

sns.pairplot(df[selected_features + ['Genre']], hue = 'Genre')
plt.show()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) 

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))
used_labels = np.unique(y_test)
used_class_names = le.inverse_transform(used_labels)
print("\nClassification Report:", classification_report(y_test, y_pred, labels=used_labels, target_names=used_class_names))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, xticklabels=used_class_names, yticklabels=used_class_names, fmt='d', cmap='Blues')
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix - Genre Tahmini")
plt.show()
































