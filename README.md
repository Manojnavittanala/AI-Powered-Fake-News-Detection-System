# 📰 Fraud News Detector

An AI-powered Fake News Detection web application built using **Machine Learning**, **Natural Language Processing (NLP)**, **Flask**, and **SQLite**. The application classifies news articles as **Real**, **Fake**, or **Uncertain** based on confidence scores.

---

## 🚀 Features

- 🔐 User Registration & Login Authentication
- 📰 Fake News Detection
- 🤖 Machine Learning Prediction using Logistic Regression
- 📊 Confidence Score & Probability
- 📜 Prediction History
- 📈 Dashboard Analytics
- 👨‍💼 Admin Panel
- 🌙 Dark Mode
- 📱 Responsive Design
- 🔒 Secure Session Management
- ⚠️ Uncertain Prediction Detection
- 📊 Interactive Charts using Chart.js

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap
- Chart.js

### Backend
- Python
- Flask
- SQLite

### Machine Learning
- Scikit-learn
- Logistic Regression
- TF-IDF Vectorizer
- NLP Text Preprocessing

---

## 📂 Project Structure

```
Fraud-News-Detector/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── static/
├── templates/
├── utils/
├── tests/
└── database.db
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-news-detector.git

cd fraud-news-detector
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 👤 Usage

1. Register a new account.
2. Login securely.
3. Open Dashboard.
4. Click **Start Detection**.
5. Paste a news article.
6. View:
   - Prediction
   - Confidence Score
   - Prediction History

---

## 📡 REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/login` | User Login |
| POST | `/api/register` | User Registration |
| POST | `/api/predict` | Predict News |
| GET | `/api/history` | Prediction History |
| GET | `/api/stats` | Dashboard Statistics |

---

## 🤖 Machine Learning Pipeline

Dataset:
- Fake.csv
- True.csv

Total Articles:
- **44,898**

Preprocessing:
- Lowercase Conversion
- URL Removal
- HTML Removal
- Stopword Removal
- Lemmatization
- Text Cleaning

Feature Extraction:
- TF-IDF Vectorizer
- 50,000 Features
- Unigram + Bigram

Classifier:
- Logistic Regression

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **99.04%** |
| Precision | **98.68%** |
| Recall | **99.55%** |
| F1 Score | **99.11%** |
| ROC-AUC | **99.93%** |

---

## ⚠️ Uncertain Predictions

Instead of forcing an incorrect prediction, the application returns **Uncertain** whenever the confidence score falls below the configured threshold.

This improves reliability for articles outside the training distribution.

---

## 🧪 Testing

Run:

```bash
pytest
```

---

## 📷 Screenshots

Add screenshots here after uploading them.

```
screenshots/
├── homepage.png
├── login.png
├── dashboard.png
├── detect.png
└── admin.png
```

---

## 🔮 Future Enhancements

- Transformer-based Fake News Detection (BERT / RoBERTa)
- Explainable AI (SHAP / LIME)
- News URL Detection
- PDF Report Generation
- Multi-language Detection
- User Profile Management
- Email Verification
- Password Reset

---

## 📄 License

This project was developed for educational purposes as a B.Tech Artificial Intelligence & Data Science project.

---

## 👩‍💻 Developer

**Manojna Vittanala**

B.Tech – Artificial Intelligence & Data Science

GitHub: https://github.com/Manojnavittanala/fraud-news-detector.git

LinkedIn: https://linkedin.com/in/manojna-vittanala-932478332