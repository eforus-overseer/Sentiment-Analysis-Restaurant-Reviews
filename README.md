## Sentiment Analysis of Restaurant Reviews

### Overview
This project focuses on performing sentiment analysis on restaurant reviews to determine whether the reviews are positive or negative. The analysis is conducted using various machine learning techniques.

### Project Workflow

1. **Importing the Libraries**
   The first step is to import the necessary libraries for data manipulation, visualization, and machine learning.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    %matplotlib inline
    ```

2. **Importing the Dataset**
   The dataset used in this project is `Restaurant_Reviews.tsv`, which contains reviews and their corresponding sentiment labels.

    ```python
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    ```

3. **Cleaning the Texts**
   Each review undergoes a text cleaning process to remove non-essential characters, stop words, and to perform stemming. This step is crucial for preparing the data for the Bag of Words model.

    ```python
    # Example of text cleaning process
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    ```

    ![Text Cleaning Visualization](path_to_cleaning_visualization.png)

4. **Creating the Bag of Words Model**
   The cleaned text data is transformed into a Bag of Words model, which converts the text data into numerical features suitable for machine learning algorithms.

    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    ```

    ![Bag of Words Model Visualization](path_to_bag_of_words_visualization.png)

5. **Splitting the Dataset**
   The dataset is split into a training set and a test set to evaluate the performance of the machine learning models.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    ```

6. **Training the Naive Bayes Model**
   A Naive Bayes model is trained on the training set and used to predict the sentiments of the reviews in the test set.

    ```python
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    ```

7. **Predicting the Test Set Results**
   The trained Naive Bayes model is used to predict the sentiments of the test set reviews.

    ```python
    y_pred = classifier.predict(X_test)
    ```

8. **Making the Confusion Matrix**
   A confusion matrix is created to evaluate the performance of the Naive Bayes model.

    ```python
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    ```

    ![Naive Bayes Confusion Matrix](path_to_naive_bayes_confusion_matrix.png)

9. **Random Forest Model**
   In addition to the Naive Bayes model, a Random Forest model is trained and evaluated for comparison.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ```

    ![Random Forest Confusion Matrix](path_to_random_forest_confusion_matrix.png)

### Conclusion
The project demonstrates the process of building a sentiment analysis model using machine learning techniques. It highlights the importance of text preprocessing and the effectiveness of different models in classifying sentiments.

### Future Work
Future improvements could include experimenting with different text vectorization techniques (e.g., TF-IDF), trying other machine learning models (e.g., SVM, LSTM), and tuning hyperparameters for better performance.
