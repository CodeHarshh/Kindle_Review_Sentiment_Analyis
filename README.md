
# Kindle Review Sentiment Analysis

This project focuses on analyzing the sentiment of Kindle product reviews. The goal is to predict whether a review is **positive** or **negative** based on the content of the review. Different techniques are used for text preprocessing and feature extraction, including **Bag-of-Words (BOW)**, **Average Word2Vec**, and **TF-IDF**. A machine learning model is then trained to classify the sentiment.

## Key Features

- **Text Preprocessing**: The reviews are cleaned by removing special characters, stopwords, URLs, and HTML tags to improve the quality of the data.
- **Vectorization Techniques**: 
  - **BOW**: A method where each review is represented by a vector of word counts.
  - **Average Word2Vec**: Words in reviews are converted into vectors using Word2Vec, and then the average vector for the entire review is computed.
  - **TF-IDF**: This method assigns weights to words based on their frequency and importance across the entire dataset.
- **Model Training**: After transforming the text into numerical features, a machine learning model is trained to classify reviews as positive or negative.
- **Model Evaluation**: The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python**: The main programming language used for this project.
- **Libraries**:
  - `pandas` for data manipulation.
  - `gensim` for Word2Vec.
  - `sklearn` for machine learning models and evaluation metrics.
  - `nltk` for text preprocessing (e.g., stopwords removal).
  - `numpy` for numerical operations.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CodeHarshh/Kindle_Review_Sentiment_Analyis.git
   cd Kindle_Review_Sentiment_Analyis
   ```

2. **Install required dependencies**:
   Use the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**:
   Once the dependencies are installed, run the script to preprocess the data, train the model, and evaluate its performance. You can modify the script to use any of the vectorization techniques (BOW, Average Word2Vec, or TF-IDF) and train the classifier on it.

4. **View Results**:
   After training, the script will output evaluation metrics like accuracy, precision, recall, and F1-score, which will help you understand how well the model is performing.

## Dataset

The dataset used contains Kindle product reviews and their sentiment labels. Each review is stored in the `reviewText` column, and the sentiment label (1 for positive and 0 for negative) is stored in the `sentiment` column.

## Example Output

After running the model, you'll get a summary of the model's performance:
```
Accuracy: 85%
Confusion Matrix:
[[50  5]
 [ 7 38]]
```
This shows the accuracy of the model, along with the confusion matrix indicating how well the model distinguishes between positive and negative reviews.
