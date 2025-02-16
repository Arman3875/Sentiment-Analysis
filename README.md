# Sentiment Analysis on Amazon Product Reviews

## 📌 Project Overview
This project focuses on analyzing Amazon product reviews to determine the sentiment of a review (Positive or Negative). The goal is to build machine learning models that can predict the sentiment of a given review based on its textual content.

## 📂 Dataset Overview
### 📄 Dataset Description
The dataset contains Amazon product reviews with the following key attributes:
- `reviewText`: The textual content of the review.
- `sentiment`: A binary label indicating sentiment:
  - `1` for positive sentiment
  - `0` for negative sentiment

### 🎯 Objective
- Perform **Exploratory Data Analysis (EDA)** to understand patterns in the dataset.
- Preprocess text data (e.g., tokenization, stopword removal, vectorization).
- Train multiple **machine learning models** to classify sentiment.
- Compare the performance of different models using evaluation metrics.

---
## ⚙️ Model Selection
### ✅ Chosen Machine Learning Models
We implemented and compared the following models:

#### **📊 Statistical Models:**
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Naïve Bayes
- Gradient Boosting (XGBoost, AdaBoost, CatBoost)

#### **🧠 Neural Networks:**
- LSTM (Long Short-Term Memory)
- GRUs (Gated Recurrent Units)

---
## 📊 Model Evaluation
### **Comparison of Model Performance**
Below is a table summarizing the performance of different models:

| Metric                  | Logistic Regression | Random Forest | SVM        | Neural Network |
|-------------------------|---------------------|--------------|------------|----------------|
| **Accuracy**            | 0.89175             | 0.87625      | 0.89550    | 0.91           |
| **Precision**           | 0.9101              | 0.8814       | 0.9067     | 0.94 (Class 1), 0.79 (Class 0) |
| **Recall**              | 0.9517              | 0.9675       | 0.9615     | 0.93 (Class 1), 0.83 (Class 0) |
| **F1 Score**            | 0.9304              | 0.9224       | 0.9333     | 0.94 (Class 1), 0.81 (Class 0) |
| **ROC-AUC**             | 0.9421              | 0.9311       | 0.9457     | N/A            |

### **🏆 Best Performing Models:**
- **Accuracy:** Neural Network (0.91)
- **Precision:** Neural Network for Class 1 (0.94)
- **Recall:** Random Forest (0.9675)
- **F1 Score:** Neural Network for Class 1 (0.94)
- **ROC-AUC:** SVM (0.95)

### **📈 Strengths & Weaknesses**
#### **Strengths:**
- **Neural Network:** High accuracy, precision, and F1 Score, making it ideal for applications prioritizing Class 1.
- **SVM:** Best overall balance across accuracy, precision, recall, and ROC-AUC.
- **Logistic Regression:** Reliable and interpretable.

#### **Weaknesses:**
- **Random Forest:** Highest recall but slightly lower precision.
- **Neural Network:** Struggles with Class 0 predictions, showing some imbalance.

### **🔍 Key Insights and Challenges**
#### **📌 Data Preprocessing:**
- **Challenges:** Handling missing values, outliers, and categorical variables.
- **Lessons Learned:** Proper text preprocessing (tokenization, stopword removal, vectorization) is crucial for better model performance.

#### **🛠 Model Training:**
- **Challenges:** Hyperparameter tuning was time-consuming.
- **Lessons Learned:** Grid Search & Random Search help optimize hyperparameters. Cross-validation ensures reliable performance estimates.

#### **📊 Model Evaluation:**
- **Challenges:** Choosing the right evaluation metrics was critical.
- **Lessons Learned:** Using multiple metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) gives a complete picture of model performance.

---
## 📌 Visualizations
### **📈 ROC Curve:**
- The ROC curve visualization showed that all models performed well, with SVM achieving the highest ROC-AUC of 0.95.

### **📑 Classification Report (Neural Network):**
- Demonstrated high precision and recall for both classes, with an overall accuracy of 0.91.

---
## ✅ Recommendations
- **Neural Network:** Best for applications prioritizing high precision & recall for positive reviews.
- **SVM:** Ideal when balance across metrics is needed.
- **Logistic Regression:** A simple and effective baseline model.
- **Random Forest:** Best for recall-focused applications.

---
## 📌 Future Work
🔹 **Improve Text Preprocessing:** Try advanced NLP techniques like word embeddings (Word2Vec, GloVe, BERT).
🔹 **Hyperparameter Tuning:** Further optimize parameters for better model performance.
🔹 **Ensemble Learning:** Experiment with ensemble methods to combine multiple models.
🔹 **Apply on a Larger Dataset:** Train models on a larger dataset to enhance generalizability.

---
## 🚀 How to Run This Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-amazon.git
   cd sentiment-analysis-amazon
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train_model.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---
## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## 📩 Contact
📧 **Arman Hossain** - armanhossainiueee@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/arman-hossain-1413991a4/)  
🔗 [GitHub](https://github.com/Arman3875/)  

---
> **Note:** This project is a learning exercise in Natural Language Processing (NLP) and sentiment analysis using machine learning techniques.
