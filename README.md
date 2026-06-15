# Customer Sentiment Analysis — Amazon Alexa Reviews
Built a sentiment classification model on 3,149 Amazon Alexa reviews using TF-IDF and NLP preprocessing; Random Forestachieved 93.2% accuracy, outperforming Logistic Regression.

## Overview
This project classifies Amazon Alexa product reviews as positive or negative using Natural Language Processing and machine learning. It covers text preprocessing, TF-IDF feature extraction, and a comparison of two classification models.

## Dataset
- Source  : Amazon Alexa Reviews Dataset
- Records : 3,150 reviews (3,149 after removing nulls)
- Columns : rating, date, variation, verified_reviews, feedback
- Labels  : feedback = 1 (positive), 0 (negative)

## Process
1. Loaded and inspected the dataset, checked for missing values
2. Cleaned review text — lowercased, removed URLs, removed punctuation/numbers, normalized whitespace
3. Mapped numeric feedback labels to readable sentiment categories
4. Split data 80/20 with stratified sampling
5. Converted text to numerical features using TF-IDF (5,000 max features, English stopwords removed)
6. Trained Logistic Regression and Random Forest classifiers
7. Tested both models on a custom example review

## Tech Stack
Python, Pandas, NumPy, Scikit-learn (TF-IDF, Logistic Regression, Random Forest), Regex for text cleaning

## Results
|        Model        | Accuracy |
|---------------------|----------|
| Logistic Regression |   92.0%  |
| Random Forest       |   93.2%  |

## Key Findings
- Random Forest outperformed Logistic Regression overall and was notably better at identifying negative reviews
- The dataset is heavily imbalanced (~92% positive reviews), which caused Logistic Regression to default toward predicting "positive" almost exclusively
- This highlights the importance of checking class distribution before relying on accuracy alone as an evaluation metric
