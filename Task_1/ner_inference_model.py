import torch
import torch.nn as nn
from transformers import BertTokenizer
from torchcrf import CRF
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Model definition
class NERBertModel(nn.Module):
    def __init__(self, num_tag, class_weights=None):
        super(NERBertModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.crf = CRF(num_tag, batch_first=True)
        self.class_weights = class_weights

    def forward(self, ids, mask, token_type_ids, target_tags=None):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[0]
        bert_out = self.bert_drop(output)
        emissions = self.out_tag(bert_out)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(emissions.device)
            emissions = emissions * class_weights
        if target_tags is not None:
            log_likelihood = self.crf(emissions, target_tags, mask=mask.byte(), reduction='mean')
            return emissions, -log_likelihood
        pred_tags = self.crf.decode(emissions, mask=mask.byte())
        return pred_tags, None

# Load saved model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tag_encoder = torch.load("tag_encoder.pth")
num_tag = len(tag_encoder.classes_)
model = NERBertModel(num_tag=num_tag).to(device)
model.load_state_dict(torch.load("ner_bert_model.pth"))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function
def predict_sentence(sentence, model, tokenizer, tag_encoder, device):
    model.eval()
    inputs = tokenizer(sentence.split(), return_tensors="pt", truncation=True, padding=True, is_split_into_words=True)
    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs.get('token_type_ids', None)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        pred_tags, _ = model(ids=input_ids, mask=mask, token_type_ids=token_type_ids)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    pred_tags = [tag_encoder.inverse_transform([pred])[0] for pred in pred_tags[0]]

    return tokens, pred_tags

# Inference demo with examples
sentences = [
    "Climbing Mount Everest is a challenge.",
    "The Annapurna range is in Nepal.",
    "Mount Kilimanjaro is in Africa.",
    "The Andes mountains span several countries.",
    "K2 is the second highest mountain in the world."
]

for sentence in sentences:
    tokens, pred_tags = predict_sentence(sentence, model, tokenizer, tag_encoder, device)
    print("Tokens:", tokens)
    print("Predicted Tags:", pred_tags)
    print()
