import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from constants import *

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@st.cache(allow_output_mutation=True)
def load_tokenizer_and_model():
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  model = AutoModel.from_pretrained(checkpoint)
  return tokenizer, model

with st.spinner('Tokenizer and Model are being loaded..'):
  tokenizer, model = load_tokenizer_and_model()

st.write("""
         # Paraphrase Detection
         """
          )

def predict(sentences, model, tokenizer):
  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)

  # Perform pooling
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  # Normalize embeddings
  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  return np.dot(sentence_embeddings[0],sentence_embeddings[1])

src = st.text_input('Enter Source sentence')
tgt = st.text_input('Enter Target sentence')
bt = st.button("Do paraphrase identification")

sentences = [src, tgt]
if (bt):
  if predict(sentences, model, tokenizer) > 0.6:
    st.success("Paraphrasing") 
  else:
    st.success("Not Paraphrasing")
