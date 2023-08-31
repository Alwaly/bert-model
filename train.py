from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from huggingface_hub import notebook_login

notebook_login()

class Data(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_len):
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df['news'][index]
        label = self.df['encoded_labels'][index]
        inputs =  self.tokenizer(text=text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        return {
            'input_ids':inputs["input_ids"].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

def dataloader(dataset, batch_size, shuffle):
    return  DataLoader(dataset=dataset, batch_size= batch_size, shuffle=shuffle)

def model_maker(model_name):
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

dataset = Data(csv_file="Spanish_News_encoded.csv", tokenizer_name='dccuchile/bert-base-spanish-wwm-uncased', max_len=120)
data_loader = dataloader(dataset, 2, True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(num_epochs, learning_rate):
    model = model_maker("dccuchile/bert-base-spanish-wwm-uncased")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    return model

if __name__=="__main__":
    model = main(5, 0.00001)
    model.push_to_hub("Alwaly/spanish-text-classification")
    tokenizer=AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    tokenizer.push_to_hub("Alwaly/spanish-text-classification")