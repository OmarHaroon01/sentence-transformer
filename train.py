from transformers import AutoModel
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FinalDataset(IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        for file_path in self.file_paths:
            tensors = torch.load(file_path)
            for input_tensor, output_tensor in zip(tensors[:-1], tensors[1:]):
                yield input_tensor, output_tensor

train_file_paths = ["train/output_5000000_5050000.pt",
                    "train/output_5050000_5070000.pt",
                    "train/output_5070000_5090000.pt",
                    "train/output_5090000_5110000.pt",
                    "train/output_5110000_5130000.pt",
                    "train/output_5130000_5150000.pt",
                    "train/output_5150000_5170000.pt",
                    "train/output_5170000_5190000.pt",
                    "train/output_5190000_5210000.pt",
                    "train/output_5210000_5230000.pt",
                    "train/output_5230000_5250000.pt",
                    "train/output_5250000_5270000.pt",
                    "train/output_5270000_5290000.pt",
                    "train/output_5290000_5310000.pt",
                    "train/output_5310000_5360000.pt",
                    "train/output_5360000_5410000.pt",
                    "train/output_5410000_5460000.pt",
                    "train/output_5460000_5500000.pt",
                    "train/output_5500000_5550000.pt",
                    "train/output_5550000_5570000.pt",
                    "train/output_5570000_5580000.pt",
                    "train/output_5580000_5630000.pt",
                    "train/output_5630000_5680000.pt",
                    "train/output_5680000_5730000.pt",
                    "train/output_5730000_5780000.pt",
                    "train/output_5780000_5830000.pt",
                    "train/output_5830000_5880000.pt",
                    "train/output_5880000_5930000.pt",
                    "train/output_5930000_5980000.pt",
                    "train/output_5980000_6000000.pt",
                    "train/output_6000000_6050000.pt",
                    "train/output_6050000_6100000.pt",
                    "train/output_6100000_6150000.pt",]

test_file_paths = ["test/output_6150000_6200000.pt",
                   "test/output_6200000_6250000.pt",
                   "test/output_6250000_6300000.pt",
                   "test/output_6300000_6350000.pt",
                   "test/output_6350000_6400000.pt",
                   "test/output_6400000_6450000.pt",
                   "test/output_6450000_6500000.pt",
                   ]

train_dataset = FinalDataset(train_file_paths)
test_dataset = FinalDataset(test_file_paths)

batch_sz = 4
learning_rate = 0.001

train_loader = DataLoader(train_dataset, batch_size=batch_sz)
test_loader = DataLoader(test_dataset, batch_size=batch_sz)

model = AutoModel.from_pretrained("t5-base")
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the tensors from the file
loaded_data = torch.load("train/extra_tokens.pt")

# Access the tensors from the loaded data
eos_input_embedding = loaded_data['eos']
pad_input_embedding = loaded_data['pad']

train_loader_size = 0
test_loader_size = 0
for epoch in range(1):
  epoch_loss = 0
  cnt = 0
  for inputs, targets in train_loader:

    train_loader_size += batch_sz

    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)

    with torch.no_grad():
      llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
      llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 1, :] = targets

    inputs = inputs.unsqueeze(1)
    llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = llmm_label_embedding_final)
    loss = criterion(llmm_output[0], llmm_label_embedding_final)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    cnt = cnt + 1

    if cnt % 15000 == 0:
      checkpoint = {
          'epoch': epoch,
          'model': model.state_dict(),
          'train_loss': epoch_loss,
          'optimizer': optimizer.state_dict()}
      torch.save(checkpoint, 'T5_model_NSP.pth')  # Save model in current directory
      current_directory = os.getcwd()
      with open(os.path.join(current_directory, 'step_loss.txt'), 'a') as file:
          file.write(f'Epoch: {epoch+1}, Step: {cnt}, Loss: {loss}\n')
      print(loss)

  valid_loss = 0

  for inputs, targets in test_loader:

    test_loader_size += batch_sz


    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)


    with torch.no_grad():
      llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
      llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 1, :] = targets

      inputs = inputs.unsqueeze(1)


      llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = llmm_label_embedding_final)
      loss_func = criterion(llmm_output[0], llmm_label_embedding_final)


    loss = loss_func.item()
    valid_loss += loss


  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss.txt'), 'a') as file:
    file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')



progress_bar = tqdm(range(9 * (test_loader_size + train_loader_size)))
for epoch in range(1, 10):
  epoch_loss = 0
  cnt = 0
  for inputs, targets in train_loader:

    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)

    with torch.no_grad():
      llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
      llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 1, :] = targets

    inputs = inputs.unsqueeze(1)
    llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = llmm_label_embedding_final)
    loss = criterion(llmm_output[0], llmm_label_embedding_final)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    cnt = cnt + 1

    if cnt % 15000 == 0:
      checkpoint = {
          'epoch': epoch,
          'model': model.state_dict(),
          'train_loss': epoch_loss,
          'optimizer': optimizer.state_dict()}
      torch.save(checkpoint, 'T5_model_NSP.pth')  # Save model in current directory
      current_directory = os.getcwd()
      with open(os.path.join(current_directory, 'step_loss.txt'), 'a') as file:
          file.write(f'Epoch: {epoch+1}, Step: {cnt}, Loss: {loss}\n')
      print(loss)

    progress_bar.update(batch_sz)


  valid_loss = 0

  for inputs, targets in test_loader:

    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)


    with torch.no_grad():
      llmm_label_embedding_final = torch.empty(batch_sz, 3, 768).to(DEVICE)
      llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)
      llmm_label_embedding_final[:, 1, :] = targets

      inputs = inputs.unsqueeze(1)


      llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = llmm_label_embedding_final)
      loss_func = criterion(llmm_output[0], llmm_label_embedding_final)

    loss = loss_func.item()
    valid_loss += loss

    progress_bar.update(batch_sz)

  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss.txt'), 'a') as file:
    file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}\n')

progress_bar.n = (9 * (test_loader_size + train_loader_size))
progress_bar.refresh()

checkpoint = {
  'model': model.state_dict(),
  'optimizer': optimizer.state_dict()}
torch.save(checkpoint, 'T5_model_NSP.pth')  # Save model in current directory
