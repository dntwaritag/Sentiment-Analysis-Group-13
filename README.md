# Sentiment Analysis of IMDb Movie Reviews

---

## Overview

This project explores and compares various machine learning and deep learning models for sentiment analysis on the IMDb 50K Movie Reviews dataset. Our goal is to classify movie reviews as either positive or negative, evaluating the performance of both traditional machine learning approaches and advanced deep learning architectures.

---

## Dataset

We utilized the **IMDb 50K Movie Reviews dataset**, a widely used benchmark for sentiment analysis. This dataset comprises 50,000 highly polar movie reviews, split equally into 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0).

**Key Features of the Dataset:**
* **Size:** 50,000 movie reviews
* **Labels:** Binary (positive/negative)
* **Source:** User-generated movie reviews from IMDb

You can find the dataset [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

---

## Repository Structure

Our repository is organized as follows:

```
├── individual_notebooks/
│   ├── sentiment_analysis_lindah_nyambura.ipynb
│   ├── SentimentAnalysis_Assignment_MKK.ipynb
│   ├── member3_notebook.ipynb
│   └── member4_notebook.ipynb
├── sentiment_analysis.ipynb
├── README.md
└── LICENSE
```

* **`individual_notebooks/`**: This folder contains individual Jupyter notebooks developed by each group member, showcasing their specific model implementations, experiments, and findings.
* **`sentiment_analysis.ipynb`**: This is the main project notebook. It encompasses the following:
    * **Exploratory Data Analysis (EDA)**: Insights into the dataset, including word distributions, sentiment balance, and common terms.
    * **Data Preprocessing**: Steps taken to clean and prepare the text data for model training (e.g., tokenization, stop word removal, stemming/lemmatization).
    * **Best Performing Traditional Models**: Implementation and evaluation of the top-performing Logistic Regression and Support Vector Machine (SVM) models.
    * **Best Performing Deep Learning Models**: Implementation and evaluation of the best-performing deep learning models contributed by each member.
* **`README.md`**: This file, providing an overview of the project.
* **`LICENSE`**: The license under which this project is distributed.

---

## Models Explored

Our project explores a diverse set of models, comparing the strengths of traditional machine learning with the capabilities of deep learning.

### Traditional Machine Learning Models

We investigated the following traditional models:

* **Logistic Regression**: A linear model used for binary classification, often serving as a strong baseline.
* **Support Vector Machine (SVM)**: A powerful model that finds an optimal hyperplane to separate data points into different classes.


### Deep Learning Models

Our deep learning exploration included:

* **Long Short-Term Memory (LSTM)**: A specialized type of RNN capable of learning long-term dependencies, particularly effective for sentiment analysis due to its ability to remember important information over extended sequences.
    * **LSTM with GloVe Embeddings and Attention**: This advanced LSTM variant leverages pre-trained GloVe word embeddings for richer semantic representations and incorporates an attention mechanism to allow the model to focus on the most relevant parts of the input sequence.
* **Gated Recurrent Unit (GRU)**: A simplified version of LSTM, offering similar performance with fewer parameters, making it computationally more efficient.
    * **Bidirectional GRU**: This GRU variant processes the input sequence in both forward and backward directions, capturing context from both past and future words, which can significantly improve understanding of sentiment.

---

## Group Contributions

This project was a collaborative effort. Here's a breakdown of individual contributions to model training and exploration:

* **Kenny Kevin Mugisha:**
    * **Traditional Models:** Logistic Regression, Support Vector Machine (SVM)
    * **Deep Learning Models:** LSTM with GloVe Embeddings and Attention
* **Lindah Nyambura:**
    * **Traditional Models:** Logistic Regression, Support Vector Machine (SVM)
    * **Deep Learning Models:** Bidirectional GRU
* **Fidel Impano:**
    * **Traditional Models:** Logistic Regression
* **Denys Ntwaritaganzwa :**
    * *(To be updated: Please fill in Member 4's contributions here)*

---

## Setup and Usage

To run the notebooks and reproduce our results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all necessary libraries in your environment.)*
4.  **Download the IMDb 50K Movie Reviews dataset.** Place it in a designated `data/` folder (or adjust paths in notebooks accordingly).
5.  **Open the Jupyter notebooks:**
    ```bash
    jupyter notebook
    ```
    You can then navigate to `sentiment_analysis.ipynb` or individual notebooks within `individual_notebooks/` to explore the code and results.

---

## Future Work

* Experiment with other deep learning architectures (e.g., Transformers, BERT-based models).
* Conduct hyperparameter tuning more extensively for all models.
* Investigate different word embedding techniques beyond GloVe (e.g., Word2Vec, FastText).
* Implement explainable AI techniques to understand model predictions better.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or inquiries, please contact the project team.
