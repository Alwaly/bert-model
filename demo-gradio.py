import gradio as gr
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from huggingface_hub import login
#login()

import gradio as gr
[3, 4, 0, 2, 1, 6, 5]
classes = dict({
    "3" : 'Otra', 
    "4" : 'Regulaciones', 
    "0" : 'Alianzas', 
    "2" : 'Macroeconomia', 
    "1" : 'Innovacion',
    "6" : 'Sostenibilidad', 
    "5" : 'Reputacion'
})
tokenizer = AutoTokenizer.from_pretrained("Alwaly/spanish-text-classification")
model = BertForSequenceClassification.from_pretrained("Alwaly/spanish-text-classification", num_labels=7)
def predict(text):
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        output = model(**input)
        pred = torch.max(output.logits, dim=1)
        indice = pred.indices.item()
        print(indice)
    return classes[f"{indice}"]

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
    
demo.launch()   