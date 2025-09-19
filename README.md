# 📝 Minecraft Review Sentiment Analysis with Deep Learning

## This project is a comprehensive implementation of sentiment analysis on Minecraft game reviews sourced from the Google Play Store. It classifies user reviews into positive, negative, or neutral sentiments using deep learning models built with TensorFlow and Keras. The entire workflow is covered, from data scraping and extensive text preprocessing to model training, evaluation, and inference.


## ✨ Key Features
- Automated Data Scraping: A dedicated script (scraping_playstore.ipynb) fetches up to 10,000 recent game reviews directly from the Google Play Store for the Indonesian region.
- Advanced Text Preprocessing: Implements a robust preprocessing pipeline tailored for Indonesian user-generated text, including:
  - Text Cleaning: Removes noise such as URLs, punctuation, numbers, and emojis.
  - Slang Word Normalization: Converts informal slang into its formal Indonesian equivalent using a custom dictionary (kamus_slang.csv).
  - Stopword Removal: Filters out common words that carry little semantic weight using a custom NLTK stopword list.
  - Stemming: Reduces words to their root form using the PySastrawi library to standardize the vocabulary.
- Class Imbalance Handling: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to create a balanced dataset, preventing model bias towards the majority class (positive reviews).
- Dual-Scheme Model Architecture: Develops and compares two distinct deep learning approaches for sentiment classification:
  - Dense Neural Network with TF-IDF: A classic but powerful approach combining TF-IDF for feature extraction with a multi-layer dense neural network.
  - Bidirectional LSTM with Word2Vec: A sequential model that leverages custom Word2Vec embeddings to capture contextual relationships between words in a review.
- Comprehensive Evaluation: Models are rigorously evaluated based on accuracy, precision, recall, and F1-score. The evaluation is performed on different training/testing data splits (80/20 and 70/30) to ensure robustness.
- Inference Function: Provides a ready-to-use function to predict the sentiment of new, unseen review text.


## 🚀 Quick Start
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab

### 1. Clone the Repository
`git clone https://github.com/your-username/Minecraft-Review-Sentiment-Analysis.git` <br>
`cd Minecraft-Review-Sentiment-Analysis` <br>

### 2. Install Dependencies
Install all the required libraries using pip. You can run the installation command from the first cell of the Sentimen_Analisis_Minecraft.ipynb notebook or create a requirements.txt file. <br>
`pip install pandas scipy gensim tensorflow scikit-learn nltk imblearn emoji PySastrawi google-play-scraper` <br>

### 3. Run the Notebooks
1. (Optional) Scrape Fresh Data: Open and run the `scraping_playstore.ipynb` notebook to collect the latest reviews. This will generate a new `hasil_ulasan_minecraft.csv` file. <br>
2. Train and Evaluate Models: Open and run all cells in the `Sentimen_Analisis_Minecraft.ipynb` notebook to perform preprocessing, data balancing, model training, and evaluation. <br>


## 📊 Results and Evaluation
The models were trained and evaluated on two different data splits. The Bidirectional LSTM with Word2Vec model consistently achieved the highest accuracy on the testing set, demonstrating its superior ability to understand text context. <br>

| Model Architecture | Data Split (Train/Test) | Training Accuracy | Testing Accuracy |
| :--- | :---: | :---: | :---: |
| Dense + TF-IDF | 80% / 20% | 95.10% | 89.79% |
| **Bi-LSTM + Word2Vec** | **80% / 20%** | **95.11%** | **91.84%** |
| Dense + TF-IDF | 70% / 30% | 95.32% | 89.06% |


## 🛠️ Technologies & Libraries
- Core Libraries: Python 3, TensorFlow, Keras
- Data Manipulation: Pandas, NumPy
- Text Processing: Scikit-learn (TfidfVectorizer), NLTK, Gensim (Word2Vec), PySastrawi, Emoji
- Data Scraping: google-play-scraper
- Data Balancing: imbalanced-learn (SMOTE)
- Environment: Jupyter Notebook / Google Colab


## ⭐ Final Result: 5 Stars (Maximum Rating)
![WhatsApp Image 2025-09-19 at 09 42 11_dcd826fd](https://github.com/user-attachments/assets/9e5627dc-5288-4bdd-ac0e-1c24e710a14f)

This project successfully passed all submission criteria, including mandatory and optional requirements, achieving the highest possible rating of 5 stars.

**Reviewer's Note for Future Improvement:**
While the project successfully met all core and optional criteria, the reviewer provided several valuable suggestions for future enhancements. These recommendations focus on improving code efficiency, preprocessing techniques, and the robustness of the evaluation process.

Code Quality & Efficiency:
- Refactor Repetitive Code: Encapsulate recurring code blocks, such as the inference logic and visualization generation, into functions to improve reusability and maintainability.
- Streamline Workflow with Pipelines: Implement Scikit-learn or TensorFlow Pipelines to automate the end-to-end workflow from preprocessing to model evaluation, ensuring a more efficient and less error-prone process.
- Optimize Environment Setup: Refine the requirements.txt file to only include essential libraries used directly in the project. Tools like pipreqs or pipreqsnb can be used to generate a clean dependency list automatically.

Data Preprocessing & Augmentation:
- Full Utilization of Slang Dictionary: Although a slang dictionary (kamus_slang.csv) was created, it was not fully implemented in the preprocessing pipeline. Integrating it would significantly improve the normalization of informal language.
- Data Augmentation: Enrich the dataset by applying text augmentation techniques like synonym replacement, back-translation, or random insertion to improve model generalization.
- Preprocessing Order: Ensure text preprocessing is performed before sentiment labeling to prevent potential mislabeling due to noise or special characters in the raw text.

Model Evaluation & Robustness:
- Implement Cross-Validation: Use k-fold cross-validation during training to get a more reliable measure of model performance and to ensure it generalizes well to unseen data, mitigating the risk of overfitting.
- Visualize with Confusion Matrix: Add a confusion matrix for each model to provide a clear, visual breakdown of its performance across all classes (positive, negative, neutral), complementing metrics like accuracy and F1-score.
- Conduct In-Depth Error Analysis: Manually review misclassified samples to identify patterns and weaknesses in the model, such as difficulties with sarcasm, complex negations, or nuanced slang.

Interpretability & Experiment Tracking:
- Enhance Model Interpretability: Apply XAI (Explainable AI) frameworks like SHAP or LIME to understand which words or phrases most significantly influence the model's predictions.
- Use Experiment Tracking Tools: Adopt tools like MLflow or Weights & Biases to systematically log experiments, track hyperparameters, and compare model results, leading to a more organized research process.


## ⭐ Special Thanks
A big thank you to Dicoding for providing an excellent and challenging learning platform. This project was an invaluable experience, pushing me to master complex concepts in natural language processing and deep learning, from text preprocessing and feature extraction to building and evaluating sophisticated models like Bidirectional LSTMs.


## ⚠️ Disclaimer
This repository is the result of a final project submission for the "Belajar Fundamental Deep Learning" course at Dicoding Academy. Please use this code as a reference for learning purposes only.

**Do not copy and paste this project for your own submission**. Plagiarism is strictly prohibited by Dicoding and will be detected.
