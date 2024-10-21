import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from torchcrf import CRF
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

# Data loading and preprocessing
data = pd.read_csv("ner_dataset_mountains.csv")
data.dropna(subset=['word', 'tag'], inplace=True)
data = data[data['word'] != ',']

# Assign sentence_id
sentence_id = 0
sentence_ids = []
for word in data['word']:
    sentence_ids.append(sentence_id)
    if word == '.':
        sentence_id += 1
data['sentence_id'] = sentence_ids

# Group by sentence
data_gr = data.groupby("sentence_id").agg({'word': list, 'tag': list}).reset_index()

# Encode tags
tag_encoder = LabelEncoder()
data['tag'] = tag_encoder.fit_transform(data['tag'])
data_gr['tag'] = data.groupby("sentence_id")['tag'].apply(list).reset_index(drop=True)

# Split into train and validation sets
train_sent, val_sent, train_tag, val_tag = train_test_split(data_gr['word'], data_gr['tag'], test_size=0.1, random_state=10)
train_sent, train_tag = train_sent.tolist(), train_tag.tolist()
val_sent, val_tag = val_sent.tolist(), val_tag.tolist()

# Configuration class
class Config:
    CLS = [101]
    SEP = [102]
    VALUE_TOKEN = [0]
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32
    EPOCHS = 3
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    USE_CRF = True

# Dataset class for handling tokenization
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        tags = self.tags[index]
        ids, target_tag = [], []

        for i, s in enumerate(texts):
            inputs = Config.TOKENIZER.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend(input_len * [tags[i]])

        ids = ids[:Config.MAX_LEN - 2]
        target_tag = target_tag[:Config.MAX_LEN - 2]
        ids = Config.CLS + ids + Config.SEP
        target_tags = Config.VALUE_TOKEN + target_tag + Config.VALUE_TOKEN

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = Config.MAX_LEN - len(ids)
        ids += [0] * padding_len
        target_tags += [0] * padding_len
        mask += [0] * padding_len
        token_type_ids += [0] * padding_len

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tags": torch.tensor(target_tags, dtype=torch.long)
        }

# Model with optional CRF
class NERBertModel(nn.Module):
    def __init__(self, num_tag, class_weights=None):
        super(NERBertModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.crf = CRF(num_tag, batch_first=True) if Config.USE_CRF else None
        self.class_weights = class_weights

    def forward(self, ids, mask, token_type_ids, target_tags=None):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[0]
        bert_out = self.bert_drop(output)
        emissions = self.out_tag(bert_out)

        if self.class_weights is not None:
            class_weights = self.class_weights.to(emissions.device)
            emissions = emissions * class_weights

        if target_tags is not None:
            if self.crf:
                log_likelihood = self.crf(emissions, target_tags, mask=mask.byte(), reduction='mean')
                return emissions, -log_likelihood
            else:
                loss_fn = FocalLoss(alpha=0.5, gamma=2)
                return emissions, loss_fn(emissions.view(-1, self.num_tag), target_tags.view(-1))

        if self.crf:
            pred_tags = self.crf.decode(emissions, mask=mask.byte())
            return pred_tags, None
        return emissions, None

# Class weights calculation
def calculate_class_weights(train_data_loader, num_tag):
    all_tags = [tag for sample in train_data_loader.dataset for tag in sample['target_tags'].tolist()]
    tag_counts = Counter(all_tags)
    total_tags = sum(tag_counts.values())
    class_weights = {tag: total_tags / (len(tag_counts) * count) for tag, count in tag_counts.items()}
    class_weights_list = [class_weights[i] for i in range(num_tag)]
    return torch.tensor(class_weights_list, dtype=torch.float)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, target):
        ce_loss = self.ce_loss(logits, target)
        valid_mask = target != self.ignore_index
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss[valid_mask].mean()

# Training function
def train_fn(train_data_loader, model, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for data in tqdm(train_data_loader, total=len(train_data_loader)):
        for key, value in data.items():
            data[key] = value.to(device)
        optimizer.zero_grad()
        _, loss = model(ids=data['ids'], mask=data['mask'], token_type_ids=data['token_type_ids'], target_tags=data['target_tags'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return model, total_loss / len(train_data_loader)

# Validation function
def val_fn(val_data_loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(val_data_loader, total=len(val_data_loader)):
            for key, value in data.items():
                data[key] = value.to(device)
            _, loss = model(ids=data['ids'], mask=data['mask'], token_type_ids=data['token_type_ids'], target_tags=data['target_tags'])
            total_loss += loss.item()
    return total_loss / len(val_data_loader)

# Main training script
device = "cuda" if torch.cuda.is_available() else "cpu"
num_tag = len(tag_encoder.classes_)
class_weights_tensor = calculate_class_weights(DataLoader(Dataset(train_sent, train_tag)), num_tag)
model = NERBertModel(num_tag=num_tag, class_weights=class_weights_tensor).to(device)

optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "gamma", "beta"])], "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "gamma", "beta"])], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=int(len(train_sent) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS))

train_dataset = Dataset(train_sent, train_tag)
val_dataset = Dataset(val_sent, val_tag)
train_data_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE)
val_data_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE)

for epoch in range(Config.EPOCHS):
    model, train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
    val_loss = val_fn(val_data_loader, model, device)
    print(f"Epoch: {epoch + 1}, Train_loss: {train_loss:.7f}, Val_loss: {val_loss:.7f}")

# Save model and encoder
torch.save(model.state_dict(), "ner_bert_model.pth")
torch.save(tag_encoder, "tag_encoder.pth")
