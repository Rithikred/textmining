Hello, 
Thank You For taking your time to take a lot at my project. This is a project is abour Poetry 
classsification using different machine learning methods and finding the best model
to this kind of problem.

Poetry Genre Classification

Introduction

Poetry is a unique form of artistic expression that captures the essence of human emotions and experiences. 
Despite its rich cultural significance, poetry remains relatively uncharted in the realm of computational analysis. 
This project explores the classification of poems into different genres using various data analysis techniques and machine learning algorithms.

Motivation

The primary motivation for this project is to integrate technology with traditional poetic forms 
to uncover patterns and meanings within poetry. By leveraging advanced computational methodologies, we aim to enhance our understanding of poetic genres.
This project specifically focuses on genre-based classification, providing insights into how poems can be categorized based on their thematic content.

Dataset Description

The dataset used in this project is sourced from Kaggle, provided by the Poetry Foundation.
Established in 2003, the Poetry Foundation aims to discover and celebrate the best poetry and make it accessible to a wide audience.
The dataset comprises 509 unique poems categorized into three genres: Love, Mythology & Folklore, and Nature. Each poem is associated with its author and thematic subjects.

Research Questions

1. How can different text representation techniques be applied to poetry for genre classification?
2. Which classifier algorithms are most effective for categorizing poems into their respective genres?

Methods and Implementation

1. Text Preprocessing

Preprocessing involved cleaning the dataset by removing irrelevant characters, symbols, and stop words to enhance data quality.

2. Tokenization

The poems were tokenized into individual words or tokens using the NLTK library to create a structured representation of the text data.

3. Text Representation

3.1 TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF was used to transform tokenized poems into numerical vectors, highlighting the importance of words 
within each document relative to their frequency across all poems.

3.2 Word Embedding Techniques

##### 3.2.1 Word2Vec

Word2Vec was employed to capture semantic relationships between words by converting them into vectors in
a high-dimensional space. The `gensim` library was utilized for this purpose.

3.2.2 GloVe (Global Vectors for Word Representation)

GloVe embeddings were used to generate vector representations based on global statistical information, capturing semantic
and contextual relationships within the poetry.

4. Training and Testing

The dataset was split into training and testing sets. Different text representations (TF-IDF, Word2Vec, GloVe) were used 
to evaluate the performance of classifier algorithms.

5. Classifier Algorithms

5.1 Logistic Regression

A linear classification algorithm applied to predict poem genres based on transformed textual features. Implemented using the `scikit-learn` library.

5.2 Support Vector Classifier (SVC)

A non-linear classification algorithm designed to capture complex patterns in textual data. Implemented using the `scikit-learn` library.

6. Evaluation Metrics

Model performance was evaluated using accuracy scores, providing a quantitative measure of the model's ability to correctly classify poem genres.

Results and Analysis

Accuracy Scores

1. TF-IDF + Logistic Regression**: Accuracy: 77%
2. TF-IDF + SVC**: Accuracy: 76%
3. Word2Vec + Logistic Regression**: Accuracy: 76%
4. Word2Vec + SVC**: Accuracy: 75%

Key Findings

1. TF-IDF + Logistic Regression** achieved the highest accuracy (77%), indicating effective classification of poem genres.
2. Word2Vec** exhibited variable performance, especially in classifying underrepresented genres like "Mythology & Folklore."
3. Class Imbalance**: Insufficient data for certain genres, notably "Mythology & Folklore," affected classification performance.

Challenges and Improvements

1. Insufficient Data**: Underrepresented genres led to poor model performance.
2. Class Imbalance**: Addressing class imbalances through data augmentation or sampling techniques could improve results.
3. Advanced Techniques**: Exploring pre-trained language models and semantic analysis techniques could further enhance classification accuracy.

Future Work

1. Dataset Expansion**: Increase the number of samples for underrepresented genres.
2. Hyperparameter Tuning**: Fine-tune model parameters for better performance.
3. Advanced Models**: Investigate semantic analysis and pre-trained models for improved genre classification.

Conclusion

This project demonstrates the potential of computational methods in the analysis and classification of poetry. By leveraging TF-IDF, Word2Vec, 
and various classifiers, we gained insights into the strengths and limitations of different approaches. Future advancements in computational linguistics
and dataset improvements offer exciting opportunities for further exploration in the realm of poetic expression.

