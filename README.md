# Sentiment Analysis Using BERT

This project demonstrates the use of **BERT (Bidirectional Encoder Representations from Transformers)** for sentiment analysis on app reviews. It leverages deep learning techniques to classify user reviews as **negative**, **neutral**, or **positive** based on the content, showcasing the ability to apply NLP models to real-world text classification tasks.

## Key Highlights

- **State-of-the-Art NLP Model**: Utilizes `BERT-base-cased`, a powerful transformer model from Hugging Face, known for its effectiveness in various NLP tasks.
- **Fine-Tuned for Sentiment Classification**: The model is specifically fine-tuned on a labeled dataset of app reviews, allowing it to effectively discern between various sentiments expressed by users.
- **High Accuracy & Performance**: Achieved a training accuracy of around **96%** and test accuracy of **~86%** after just 5 epochs, indicating strong performance on sentiment prediction.

## Project Structure

- **`Sentiment_Analysis.ipynb`**: A Jupyter Notebook that walks through the steps of data loading, preprocessing, model training, evaluation, and visualization.
- **`reviews.csv`**: Dataset containing user reviews and associated metadata like ratings, usernames, review content, and app version.
- **`best_model_state.bin`**: Model weights saved after training to allow for quick inference on new data.

## Dataset

The dataset contains over **15,000 app reviews** labeled with user-provided ratings from an app store. Reviews are classified into three sentiment categories:
- **Negative** (ratings ≤ 2)
- **Neutral** (rating = 3)
- **Positive** (ratings ≥ 4)

The dataset is split into training, validation, and testing sets to ensure robust model evaluation.

## Model Architecture

The sentiment classifier is built on top of `BERT` and consists of:
1. **BERT Layer**: Pretrained transformer that extracts meaningful features from the review text.
2. **Dropout Layer**: Regularization layer to prevent overfitting during training.
3. **Fully Connected Layer**: Final layer that maps BERT outputs to sentiment classes (**negative**, **neutral**, **positive**).

- **Optimizer**: `AdamW`, with weight decay for regularization.
- **Learning Rate Scheduler**: Linear warm-up and decay for efficient training.
- **Loss Function**: Cross-Entropy Loss for multi-class classification.

## Installation & Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/SAI_ADITH/Sentiment-Analysis-BERT.git
    cd Sentiment-Analysis-BERT
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not available:
    ```bash
    pip install numpy pandas matplotlib scikit-learn torch transformers tqdm
    ```
3. **Install Jupyter Notebook**:
    ```bash
    pip install notebook
    ```

## How to Run the Project

1. **Open Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2. **Load `Sentiment_Analysis.ipynb`**:
   - Follow the notebook cells to preprocess data, train the model, and visualize results.
   - Ensure `reviews.csv` is in the correct directory.

3. **Make Predictions**:
   - Once trained, the model can be used to predict the sentiment of new reviews.
   - Simply modify the text in the relevant cell of the notebook to analyze custom text.

## Performance Metrics & Results

The model's performance is evaluated using standard metrics:
- **Accuracy**: ~96% on training data, ~86% on test data.
- **Precision, Recall, F1-Score**: Calculated for each sentiment class, indicating the model’s effectiveness in correctly identifying each sentiment.
- **Confusion Matrix**: Visualized to understand the model’s misclassification behavior across different classes.

## Visualizations & Insights

- **Token Length Distribution**: Analyzes the token count per review to optimize input length.
- **Training History**: Plots accuracy and loss for both training and validation sets across epochs.
- **Confusion Matrix**: Shows a breakdown of true vs. predicted sentiments, highlighting model strengths and areas for improvement.

## Potential Improvements & Extensions

1. **Hyperparameter Tuning**: Experiment with batch size, learning rate, and dropout rate for even better performance.
2. **Data Augmentation**: Increase the dataset size by incorporating synthetic reviews or leveraging similar datasets.
3. **Advanced Models**: Explore transformer models like `RoBERTa` or `DistilBERT` for potentially faster and more accurate sentiment analysis.

## Conclusion

This project serves as a comprehensive demonstration of using cutting-edge NLP techniques for real-world sentiment analysis. With a high accuracy score and efficient processing pipeline, this classifier can be easily adapted for other text classification tasks, making it highly versatile for various applications.

Feel free to reach out for any discussions on how this project can be improved or extended for more advanced NLP tasks!
