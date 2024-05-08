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

checkpoint = torch.load('T5_model_NSP_1.pth')

model = AutoModel.from_pretrained("t5-base")
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(checkpoint['optimizer'])

loaded_data = torch.load("train/extra_tokens.pt")
eos_input_embedding = loaded_data['eos']
pad_input_embedding = loaded_data['pad']

train_file_paths = ["train/output_0_10000.pt",
                    "train/output_10000_20000.pt",
                    "train/output_20000_40000.pt",
                    "train/output_40000_70000.pt",
                    "train/output_70000_100000.pt",
                    "train/output_100000_120000.pt",
                    "train/output_120000_150000.pt",
                    "train/output_150000_180000.pt",
                    "train/output_180000_220000.pt",
                    "train/output_220000_260000.pt",
                    "train/output_260000_300000.pt",
                    "train/output_300000_350000.pt",
                    "train/output_350000_400000.pt",
                    "train/output_400000_450000.pt",
                    "train/output_450000_500000.pt",
                    "train/output_500000_550000.pt",
                    "train/output_550000_600000.pt",
                    "train/output_600000_650000.pt",
                    "train/output_650000_700000.pt",
                    "train/output_700000_750000.pt",
                    "train/output_750000_800000.pt",
                    "train/output_800000_850000.pt",
                    "train/output_850000_900000.pt",
                    "train/output_900000_950000.pt",
                    "train/output_950000_1000000.pt",
                    "train/output_2585000_2600000.pt"]



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
  torch.save(checkpoint, 'T5_model_NSP_2.pth')
  
  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss_2.txt'), 'a') as file:
    file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')

checkpoint = {
  'model': model.state_dict(),
  'optimizer': optimizer.state_dict()}
torch.save(checkpoint, 'T5_model_NSP_2.pth')
