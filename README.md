# 📧 Spam Email Classifier (NLP)

This project builds an ML model to classify emails as Spam or Ham (Not Spam) using NLP techniques.

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Jupyter Notebook

## 📁 Dataset
[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Place the downloaded spam.csv in the `data/` folder.

## ⚙️ How to Run

```bash
pip install -r requirements.txt
cd src
python train_model.py
```

To test prediction:
```bash
python predict.py
```

To run the Jupyter notebook:
```bash
jupyter notebook ../notebooks/spam_classifier.ipynb
```

For Streamlit app:
```bash
streamlit run app.py
```

## 🧠 Model Performance
- Accuracy: ~98.7%
- F1 Score: ~0.97

Confusion Matrix:

<img src="images/confusion_matrix.png" width="400"/>

## ✅ Features

- TF-IDF Vectorization
- Naive Bayes Classifier
- Cleaned and tokenized input
- (Optional) Flask/Streamlit UI

---

## 💡 Optional Add-ons

| Feature            | Tool                |
|--------------------|---------------------|
| Web UI             | Streamlit / Flask   |
| Deployment         | Render / Heroku     |
| Model Versioning   | MLflow / DVC        |
| Testing            | PyTest              |

---

## ✅ Next Steps

- Try new models (Logistic Regression, Random Forest)
- Add more preprocessing
- Deploy with Flask/Streamlit
