import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training Dataset
data = {
    "subject": [
        "Urgent action required verify your account",
        "Your account has been suspended immediately",
        "Congratulations you won a prize",
        "Meeting schedule for tomorrow",
        "Invoice for your recent purchase",
        "Update your password now",
        "Limited time offer claim now",
        "Project discussion reminder"
    ],
    "label": [
        "phishing",
        "phishing",
        "phishing",
        "legitimate",
        "legitimate",
        "phishing",
        "phishing",
        "legitimate"
    ]
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["subject"])
y = df["label"]

# Model Training
model = MultinomialNB()
model.fit(X, y)

# Urgency Keywords
urgent_words = [
    "urgent", "immediately", "now",
    "verify", "suspended", "limited"
]

# Prediction Function
def detect_email(subject):
    subject_vector = vectorizer.transform([subject])
    prediction = model.predict(subject_vector)[0]

    urgency = "Normal"
    for word in urgent_words:
        if word in subject.lower():
            urgency = "High"
            break

    print("Email Subject:", subject)
    print("Phishing Detection:", prediction.upper())
    print("Urgency Level:", urgency)

# User Input
while True:
    email_subject = input("Enter email subject (or type 'exit'): ")
    if email_subject.lower() == "exit":
        break
    detect_email(email_subject)