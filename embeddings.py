from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
# Sentences we want sentence embeddings for
sentences = ["The quick brown fox jumps over the lazy dog.",
             "Now is the time for all good men to come to the aid of their country."]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    print(model_output.keys())
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)

print(tokenizer.decode(model_output.logits.argmax(dim=-1)))
embeddings = model_output.last_hidden_state[:, 0]

