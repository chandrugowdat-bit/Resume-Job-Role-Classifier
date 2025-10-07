# Step 1: Import pandas library
import pandas as pd

# Step 2: Load the dataset
data = pd.read_csv(r"C:\Users\Sahana\OneDrive\Documents\resumes.csv")

# Step 3: View first few rows
print(data.head(10))

# Step 4: Text Preprocessing
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the Resume_Text column
data['Processed_Text'] = data['Resume_Text'].apply(preprocess)

# View the processed text
print(data[['Resume_Text', 'Processed_Text']])

# Step 8: Build the Job Role Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Processed_Text'])
y = data['Job_Role']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Test with a new resume
sample_resume = "Experienced in Python, machine learning, and data visualization"
sample_processed = preprocess(sample_resume)
sample_vector = vectorizer.transform([sample_processed])
prediction = model.predict(sample_vector)
print(f"Predicted Job Role: {prediction[0]}")

# Step 9: Create a simple GUI using Tkinter
import tkinter as tk
from tkinter import messagebox

def predict_job():
    resume_text = text_box.get("1.0", tk.END)  # get text from text box
    processed = preprocess(resume_text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)
    messagebox.showinfo("Prediction", f"Predicted Job Role: {prediction[0]}")

# Create main window
window = tk.Tk()
window.title("Resume Job Role Classifier")
window.geometry("700x400")

# Header
header = tk.Label(window, text="Resume Job Role Classifier", font=("Arial", 18, "bold"), fg="blue")
header.pack(pady=10)

# Scrollable Text box
scrollbar = tk.Scrollbar(window)
text_box = tk.Text(window, height=15, width=80, yscrollcommand=scrollbar.set)
scrollbar.config(command=text_box.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_box.pack(pady=10)

# Predict Button
button = tk.Button(window, text="Predict Job Role", command=predict_job,
                   font=("Arial", 14, "bold"), bg="green", fg="white", padx=10, pady=5)
button.pack(pady=10)

# Footer
footer = tk.Label(window, text="Developed by: Sahana | Final Year Project", font=("Arial", 10, "italic"))
footer.pack(side=tk.BOTTOM, pady=5)

# Run the window
window.mainloop()

import pickle

# Save the trained model
with open("job_role_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and Vectorizer saved successfully!")
