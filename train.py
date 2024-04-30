import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import os
import nltk
import vec2text
import torch
import evaluate
from tqdm.auto import tqdm
from transformers import set_seed
from datasets import load_dataset
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from transformers import DefaultDataCollator
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, T5ForConditionalGeneration, AutoConfig

set_seed(42) 
nltk.download('punkt')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasets = load_dataset("wikipedia", "20220301.en")
datasets["train"] = load_dataset("wikipedia", "20220301.en", split=f"train[:3000000]")
datasets["validation"] = load_dataset("wikipedia", "20220301.en", split=f"train[-322934:]")

# Function to remove HTML Tags from paragraph
def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


#Function to clean the paragraphs
def clean_text(text):
    text = remove_html_tags(text)
    text = ' '.join(text.split())  # Remove redundant spaces
    text = re.sub(r'\n+', '\n', text) #Remove redundant new line
    text = re.sub(r'[^A-Za-z0-9\s,.!-?–]+', '', text)  # Remove special characters
    return text

datasets = datasets.map(lambda x: {'text': clean_text(x['text'])})


# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")


# This function is used to tokenize the dataset
def process_dataset(dataset):
    sentences = []

    for article in dataset['text']:
        article_sentences = sent_tokenize(article)
        sentences.extend(article_sentences)

    
    inputs = sentences[:-1]
    targets = sentences[1:]


    model_inputs = tokenizer(inputs, return_tensors="pt", max_length=128, truncation=True, padding="max_length") 
    labels = tokenizer(targets, return_tensors="pt", max_length=128, padding="max_length", truncation=True) 

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["label_attention_mask"] = labels["attention_mask"]

    return model_inputs

tokenized_dataset = datasets.map(process_dataset, batched=True, remove_columns=["text", "url", "id", "title"])


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE)

batch_sz = 2
learning_rate = 0.001


data_collator = DefaultDataCollator()

train_dataloader = torch.utils.data.DataLoader(
    tokenized_dataset["train"],
    batch_size=batch_sz,
    collate_fn=data_collator,
)

test_dataloader = torch.utils.data.DataLoader(
    tokenized_dataset["validation"],
    batch_size=batch_sz,
    collate_fn=data_collator,
)


model = AutoModel.from_pretrained("t5-base")
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# The sentence embedding of eos token is calculated
eos_tokens = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt")
eos_embedding = encoder(eos_tokens['input_ids'].to(DEVICE), eos_tokens['attention_mask'].to(DEVICE))
hidden_state = eos_embedding.last_hidden_state
eos_input_embedding = vec2text.models.model_utils.mean_pool(hidden_state.to(DEVICE), eos_tokens['attention_mask'].to(DEVICE)).to(DEVICE)

#The sentence embedding of pad token is calculated (Pad token is used as start token for T5)
pad_tokens = tokenizer(tokenizer.pad_token, add_special_tokens=False, return_tensors="pt")
pad_embedding = encoder(pad_tokens['input_ids'].to(DEVICE), pad_tokens['attention_mask'].to(DEVICE))
hidden_state = pad_embedding.last_hidden_state
pad_input_embedding = vec2text.models.model_utils.mean_pool(hidden_state.to(DEVICE), pad_tokens['attention_mask'].to(DEVICE)).to(DEVICE)

for epoch in range(10):
  epoch_loss = 0
  cnt = 0


  ### TRAINING LOOP FOR EACH EPOCH ###
  for batch in tqdm(train_dataloader, desc='Training:'):

    #Moving the tensors to the available DEVICE
    batch['input_ids'] = batch['input_ids'].to(DEVICE)
    batch['attention_mask'] = batch['attention_mask'].to(DEVICE)
    batch['labels'] = batch['labels'].to(DEVICE)
    batch['label_attention_mask'] = batch['label_attention_mask'].to(DEVICE)

    # This is used to calculate the sentence embedding for the input and labels
    with torch.no_grad():
        llmm_input = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        hidden_state = llmm_input.last_hidden_state
        llmm_input_embedding = vec2text.models.model_utils.mean_pool(hidden_state, batch['attention_mask']).to(DEVICE)

        # Defining an empty tensor to store the label along with EOS and PAD token (Sequence length of 3)
        llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
        llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
        llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)

        llmm_label = encoder(input_ids=batch['labels'], attention_mask=batch['label_attention_mask'])
        hidden_state = llmm_label.last_hidden_state
        llmm_label_embedding = vec2text.models.model_utils.mean_pool(hidden_state, batch['label_attention_mask']).to(DEVICE)

        llmm_label_embedding_final[:, 1, :] = llmm_label_embedding

    # This adds the sequence length(1) dimension to input tensor. Makes the tensor [batch_sz, 1, 768] from [batch_sz, 768]
    llmm_input_embedding = llmm_input_embedding.unsqueeze(1)

    llmm_output = model(inputs_embeds = llmm_input_embedding.to(DEVICE), decoder_inputs_embeds = llmm_label_embedding_final.to(DEVICE))
    loss = criterion(llmm_output[0], llmm_label_embedding_final)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    cnt = cnt + 1

    # SAVING AND PRINTING LOSS AFTER EVERY 20000 STEPS
    if cnt % 10000 == 0:
      checkpoint = {
          'epoch': epoch,
          'model': model.state_dict(),
          'train_loss': epoch_loss,
          'optimizer': optimizer.state_dict()}
      torch.save(checkpoint, 'T5_model_NSP.pth')  # Save model in current directory
      current_directory = os.getcwd()
      with open(os.path.join(current_directory, 'step_loss.txt'), 'a') as file:
          file.write(f'Step: {cnt}, Loss: {loss}\n')
      print(loss)


  ### TESTING AFTER EACH EPOCH ###

  valid_loss = 0
  for batch in tqdm(test_dataloader, desc='Testing:'):

    #Moving the tensors to the available DEVICE
    batch['input_ids'] = batch['input_ids'].to(DEVICE)
    batch['attention_mask'] = batch['attention_mask'].to(DEVICE)
    batch['labels'] = batch['labels'].to(DEVICE)
    batch['label_attention_mask'] = batch['label_attention_mask'].to(DEVICE)



    # This is used to calculate the sentence embedding for the input and labels
    with torch.no_grad():

        llmm_input = encoder(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
        hidden_state = llmm_input.last_hidden_state
        llmm_input_embedding = vec2text.models.model_utils.mean_pool(hidden_state, batch['attention_mask'])


        # Defining an empty tensor to store the label along with EOS and PAD token (Sequence length of 3)
        llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
        llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
        llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)


        llmm_label = encoder(input_ids=batch['labels'].to(DEVICE), attention_mask=batch['label_attention_mask'].to(DEVICE))
        hidden_state = llmm_label.last_hidden_state
        llmm_label_embedding = vec2text.models.model_utils.mean_pool(hidden_state, batch['label_attention_mask'])

        llmm_label_embedding_final[:, 1, :] = llmm_label_embedding

        # This adds the sequence length(1) dimension to input tensor. Makes the tensor [batch_sz, 1, 768] from [batch_sz, 768]
        llmm_input_embedding = llmm_input_embedding.unsqueeze(1)

        llmm_output = model(inputs_embeds = llmm_input_embedding.to(DEVICE), decoder_inputs_embeds = llmm_label_embedding_final.to(DEVICE))
        loss_func = criterion(llmm_output[0], llmm_label_embedding_final)

    loss = loss_func.item()
    valid_loss += loss
  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/len(train_dataloader)}, Eval Loss: {valid_loss/len(test_dataloader)}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss.txt'), 'a') as file:
    file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/len(train_dataloader)}, Eval Loss: {valid_loss/len(test_dataloader)}')

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()}
torch.save(checkpoint, 'T5_model_NSP.pth')  # Save model in current directory
