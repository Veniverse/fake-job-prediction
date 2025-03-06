# Fake Job Prediction

## Project Overview
This project aims to predict fraudulent job postings using machine learning techniques. It involves data preprocessing, feature engineering, and model training using classification algorithms.

## Dataset
- **Source**: The dataset used for this project is `fake_job_postings.csv`.
- **Cleaning**: Missing values were handled, and relevant features were extracted.
- **Feature Engineering**: Location details were refined, text data was processed, and numerical features were created.

## Data Preprocessing
- Missing values were handled by filling them with appropriate values.
- Location data was split into `state` and `city` for better analysis.
- Text features were combined to create a new column `text`.
- The dataset was balanced and cleaned for better performance.

## Exploratory Data Analysis (EDA)
- Heatmaps and count plots were generated to understand correlations and feature distributions.
- Fraudulent job postings were analyzed based on different features like `employment type`, `required experience`, and `required education`.

## Model Training
### Naive Bayes Classifier
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)
pred = nb_classifier.predict(count_test)
metrics.accuracy_score(y_test, pred)  # 97.08%
metrics.f1_score(y_test, pred)  # 0.73
```

### SGD Classifier (Logistic Regression)
```python
from sklearn.linear_model import SGDClassifier

clf_log = SGDClassifier(loss='log_loss').fit(count_train, y_train)
pred_log = clf_log.predict(count_test)
metrics.accuracy_score(y_test, pred_log)  # 97.79%
```

### SGD Classifier (Numerical Features)
```python
clf_num = SGDClassifier(loss='log_loss').fit(X_train_num, y_train)
pred_num = clf_num.predict(X_test_num)
metrics.accuracy_score(y_test, pred_num)  # 93.65%
```

### Hybrid Model (Combining Text & Numeric Predictions)
```python
prediction_array = [1 if i == 1 or j == 1 else 0 for i, j in zip(pred_num, pred_log)]
metrics.accuracy_score(y_test, prediction_array)  # 97.79%
metrics.f1_score(y_test, prediction_array)  # 0.82
```

## Model Evaluation
- **Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

cf_matrix = confusion_matrix(y_test, prediction_array)
group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = np.asarray([f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
```

## Conclusion
- The hybrid approach combining numerical and text-based predictions provided the highest accuracy of **97.79%**.
- The **Naive Bayes classifier** performed well with text data, but combining models improved the overall results.

## Future Improvements
- Experiment with deep learning models like LSTMs for better text classification.
- Use advanced NLP techniques like word embeddings (Word2Vec, GloVe) for feature extraction.
- Further optimize feature selection to reduce noise and improve model performance.

Author
Veni R
