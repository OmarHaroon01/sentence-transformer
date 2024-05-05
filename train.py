from transformers import AutoModel
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FinalDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.tensors = torch.load(file_path, map_location=torch.device('cpu'))

    def __iter__(self):
        for input_tensor, output_tensor in zip(self.tensors[:-1], self.tensors[1:]):
            yield input_tensor, output_tensor

    def __len__(self):
        return len(self.tensors) - 1  # Number of pairs of tensors

    def clear_tensors(self):
        del self.tensors
        self.tensors = None

batch_sz = 2048
learning_rate = 0.001

model = AutoModel.from_pretrained("t5-base")
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loaded_data = torch.load("train/extra_tokens.pt")
eos_input_embedding = loaded_data['eos']
pad_input_embedding = loaded_data['pad']

train_file_paths = ["train/output_3500000_3550000.pt",
                    "train/output_3550000_3600000.pt",
                    "train/output_3600000_3650000.pt",
                    "train/output_3650000_3700000.pt",
                    "train/output_3700000_3800000.pt",
                    "train/output_3800000_3900000.pt",
                    "train/output_3900000_4000000.pt",
                    "train/output_4000000_4050000.pt",
                    "train/output_4050000_4070000.pt",
                    "train/output_4070000_4120000.pt",
                    "train/output_4120000_4125000.pt",
                    "train/output_4125000_4130000.pt",
                    "train/output_4130000_4150000.pt",
                    "train/output_4150000_4200000.pt",
                    "train/output_4200000_4250000.pt",
                    "train/output_4250000_4350000.pt",
                    "train/output_4350000_4450000.pt",
                    "train/output_4450000_4500000.pt",
                    "train/output_4500000_4550000.pt",
                    "train/output_4550000_4600000.pt",
                    "train/output_4600000_4650000.pt",
                    "train/output_4650000_4700000.pt",
                    "train/output_4700000_4750000.pt",
                    "train/output_4750000_4800000.pt",
                    "train/output_4800000_4900000.pt",
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
                    "train/output_6100000_6150000.pt"]

test_file_paths = ["test/output_6150000_6200000.pt",
                   "test/output_6200000_6250000.pt",
                   "test/output_6250000_6300000.pt",
                   "test/output_6300000_6350000.pt",
                   "test/output_6350000_6400000.pt",
                   "test/output_6400000_6450000.pt",
                   "test/output_6450000_6500000.pt"]

for epoch in range(10):
  train_loader_size = 0
  test_loader_size = 0
  epoch_loss = 0
  file_num = 0
  for file_path in train_file_paths:
    train_dataset = FinalDataset(file_path)
    file_num += 1
    train_loader = DataLoader(train_dataset, batch_size=batch_sz)

    for inputs, targets in tqdm(train_loader, desc=f"Processing File {file_num} of train data"):
      inputs = inputs.to(DEVICE)
      targets = targets.to(DEVICE)
      train_loader_size += batch_sz

      with torch.no_grad():
          llmm_label_embedding_final = torch.empty(targets.size(0), 3, 768).to(DEVICE)
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

    train_dataset.clear_tensors()
  valid_loss = 0
  file_num = 0
  for file_path in test_file_paths:
    test_dataset = FinalDataset(file_path)
    file_num += 1
    test_loader = DataLoader(test_dataset, batch_size=batch_sz)

    for inputs, targets in tqdm(test_loader, desc=f"Processing File {file_num} of test data"):
      inputs = inputs.to(DEVICE)
      targets = targets.to(DEVICE)

      test_loader_size += batch_sz

      with torch.no_grad():
        llmm_label_embedding_final = torch.empty(targets.size(0), 3, 768).to(DEVICE)
        llmm_label_embedding_final[:, 0, :] = pad_input_embedding.squeeze(0)
        llmm_label_embedding_final[:, 2, :] = eos_input_embedding.squeeze(0)
        llmm_label_embedding_final[:, 1, :] = targets

        inputs = inputs.unsqueeze(1)

        llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = llmm_label_embedding_final)
        loss_func = criterion(llmm_output[0], llmm_label_embedding_final)

      loss = loss_func.item()
      valid_loss += loss

    test_dataset.clear_tensors()

  checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()}
  torch.save(checkpoint, 'T5_model_NSP.pth')
  
  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss.txt'), 'a') as file:
    file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')

checkpoint = {
  'model': model.state_dict(),
  'optimizer': optimizer.state_dict()}
torch.save(checkpoint, 'T5_model_NSP.pth')
