# Sentiment-Analysis-on-Social-Media-Dataset

This project performs sentiment analysis on tweets using the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140). It involves data preprocessing, visualization, training a Logistic Regression model using TF-IDF vectorization, and deploying the model with a simple web interface using Flask.


## 🧠 Methodology

This project focuses on building a sentiment analysis model using the Sentiment140 dataset. The goal is to classify tweets into **positive** or **negative** sentiment using machine learning. Here's a step-by-step overview of how the project was developed:

### 1. Dataset Setup
We used the Kaggle API to download the **Sentiment140** dataset, which contains 1.6 million tweets labeled for sentiment. After downloading, the dataset was unzipped and loaded into a Pandas DataFrame with the appropriate column names.

### 2. Data Cleaning & Preprocessing
Raw tweets usually contain noise like URLs, hashtags, mentions, and punctuation. We cleaned the text by:
- Removing URLs, user mentions (`@username`), hashtags, and special characters.
- Converting text to lowercase.
- Tokenizing the tweets using NLTK.
- Removing common English stopwords (e.g., "the", "and", "is").

All cleaned tweets were stored in a new column for further processing.

### 3. Label Encoding
The original dataset labels sentiment as `0` (negative) and `4` (positive). We simplified this into a binary format:
- `0` → Negative
- `4` → Positive → Mapped to `1`

### 4. Train-Test Split
We split the dataset into training and testing sets to evaluate the model's performance. This helps ensure that the model generalizes well to unseen data.

### 5. Text Vectorization
Since machine learning models can't work directly with raw text, we used **TF-IDF Vectorization** to convert the cleaned text into numerical features. We limited the vocabulary to the top 5000 terms to keep the model efficient.

### 6. Model Training
A **Logistic Regression** model was trained on the vectorized training data. Logistic Regression is a solid choice for binary classification tasks like sentiment analysis.

### 7. Evaluation
We evaluated the model on the test set using:
- **Accuracy Score**
- **Confusion Matrix** (visualized using Seaborn heatmap)
- **Classification Report** (Precision, Recall, F1-score)

These metrics gave us insights into how well the model is performing.

### 8. Saving the Model
Once we were satisfied with the model's performance, we saved:
- The trained model (`sentiment_model.pkl`)
- The TF-IDF vectorizer (`tfidf_vectorizer.pkl`)

This makes it easy to reuse or deploy the model without retraining.

### 9. Exporting for Use
Finally, both the model and vectorizer were downloaded from Google Colab so they can be used locally or integrated into a web app or API for live sentiment prediction.

---



