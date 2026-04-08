# Sentiment Analysis with LSTM in PyTorch

This project implements a **sentiment analysis web app** using a **bidirectional LSTM** in PyTorch to classify movie reviews from the **IMDB dataset** as **positive** or **negative**.  
It includes a **Streamlit UI**, **confidence visualization**.

## 🔗 Live Demo

Try the model live on Streamlit:

[Movie Sentiment Analysis LSTM Web App](https://movie-sentiment-lstm-pytorch-git-rushipatil2003.streamlit.app/)

##  Reference & Methodology

The project leverages the core concepts of LSTM (Long Short-Term Memory) networks for sequence modeling in NLP tasks. The methodology follows these steps:

* **Data Preparation** – Load the IMDB dataset, tokenize text, build vocabulary, and numerize data.
* **Model Design** – Build a bidirectional LSTM with embedding and dropout layers.
* **Training** – Use CrossEntropyLoss and Adam optimizer with mini-batches.
* **Evaluation** – Measure accuracy and loss on validation and test datasets.
* **Prediction** – Test the trained model on new sentences.

##  Key Features

- LSTM-based sentiment classifier  
- Bidirectional LSTM for better context understanding  
- Custom data loader with padding support  
- Training & evaluation loop with live metrics visualization  
- Web app interface with **mobile-friendly UI**  
- Emoji-based sentiment output & confidence bar chart  
- Quick prediction on arbitrary input text  

## 📊 Results

The model progressively learns to classify movie reviews accurately. Training and validation loss/accuracy are visualized over epochs.

Example of testing on new sentences:

| Sentence                                      | Predicted Sentiment | Probability |
|-----------------------------------------------|---------------------|------------|
| "This film is terrible!"                      | Negative            | 0.95       |
| "This film is great!"                         | Positive            | 0.92       |
| "This film is not terrible, it's great!"      | Positive            | 0.88       |
| "Not recommended"                             | Negative            | 0.91       |

> Note: The trained model is saved as `lstm.pt`.

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv lstm_env
   source lstm_env/bin/activate   # Linux/macOS
   lstm_env\Scripts\activate      # Windows

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit web app:
   ```bash
   streamlit run app.py


## Future Improvements
* Experiment with pre-trained embeddings like GloVe or BERT
* Implement attention mechanism for improved performance
* Train on more diverse datasets beyond IMDB
* Add TensorBoard or Matplotlib live plotting
* Hyperparameter tuning for better generalization
* Deployment using streamlit

## Acknowledgements
* IMDB dataset via Hugging Face datasets library
* PyTorch and TorchText documentation
* Tutorials on sequence modeling with LSTMs
