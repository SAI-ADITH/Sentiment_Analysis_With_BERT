# Sentiment Analysis Using BERT

This project uses **BERT** to classify app reviews as **negative**, **neutral**, or **positive** based on their content. The model is trained on a dataset of user reviews and fine-tuned for sentiment classification.

## Project Structure
- **`Sentiment_Analysis.ipynb`**: Code for data preprocessing, training, and evaluation.
- **`reviews.csv`**: Dataset containing app reviews and their metadata.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/Sentiment-Analysis-BERT.git
    cd Sentiment-Analysis-BERT
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is unavailable:
    ```bash
    pip install numpy pandas matplotlib scikit-learn torch transformers tqdm
    ```
3. **Install Jupyter Notebook**:
    ```bash
    pip install notebook
    ```

## Running the Code

1. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2. **Open `Sentiment_Analysis.ipynb`** and run the cells sequentially.

## Model Overview

- **Tokenizer & Model**: BERT from Hugging Face's `transformers`.
- **Architecture**: BERT model + Dropout + Fully connected layer for classification.
- **Optimizer**: `AdamW`; **Loss**: Cross-Entropy.

## Results
- Achieves **~96% training accuracy** and **~86% validation/test accuracy** after 5 epochs.
- Evaluated with precision, recall, F1-score, and confusion matrix.

---

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username. Let me know if you need any further changes!
