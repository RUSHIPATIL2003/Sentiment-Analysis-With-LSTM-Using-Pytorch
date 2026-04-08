import streamlit as st
import torch
import torch.nn as nn
import pickle
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Sentiment Movie Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------- MOBILE CSS --------
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
textarea { font-size: 16px !important; }
.stButton button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------- MODEL --------
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 n_layer, bidirectional, dropout_rate,
                 pad_index, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_index)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layer,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            output_dim
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, length,
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-1], hidden[-2]), dim=-1))
        else:
            hidden = self.dropout(hidden[-1])

        return self.fc(hidden)

# -------- LOAD --------
@st.cache_resource
def load_all():
    with open("models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    tokenizer = get_tokenizer("basic_english")

    model = LSTM(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=300,
        n_layer=2,
        bidirectional=True,
        dropout_rate=0.5,
        pad_index=vocab["<pad>"],
        output_dim=2
    )

    model.load_state_dict(torch.load("models/lstm.pt", map_location="cpu"))
    model.eval()

    return model, vocab, tokenizer

model, vocab, tokenizer = load_all()

# -------- PREDICT --------
def predict(text):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)

    if len(ids) == 0:
        return None, None

    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor, length)
        probs = torch.softmax(output, dim=-1)

    pred = torch.argmax(probs).item()
    conf = probs[0][pred].item()

    return pred, conf

# -------- UI --------
st.title(" Sentiment Movie Analyzer")
st.caption("This project implements asentiment analysis web app using a bidirectional LSTM in PyTorch to classify movie reviews from the IMDB dataset as positive or negative. It includes a Streamlit UI, confidence visualization.")

st.divider()

text = st.text_area(" Enter your review:")

if st.button(" Analyze"):
    with st.spinner("Analyzing..."):
        pred, conf = predict(text)

    if pred is None:
        st.warning(" Enter valid text")

    else:
        if pred == 1:
            st.success(f" Positive ({conf:.2f})")
            st.balloons()
        else:
            st.error(f" Negative ({conf:.2f})")

        # Chart
        probs = [conf, 1 - conf] if pred == 1 else [1 - conf, conf]

        fig, ax = plt.subplots()
        ax.bar([" Positive", " Negative"], probs)
        ax.set_ylim([0, 1])

        st.pyplot(fig)