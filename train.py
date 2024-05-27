from transformers import AutoModel, AutoConfig
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader
import math

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FinalDataset(IterableDataset):
    def __init__(self, file_path, seq_len, pad_embedding):
        self.file_path = file_path
        self.tensors_loaded = torch.load(file_path, map_location=torch.device('cpu'))
        self.tensors = torch.cat(self.tensors_loaded, dim = 0)
        self.pad_embedding = pad_embedding
        self.seq_len = seq_len

    def __iter__(self):
        tensor = self.tensors
        pad_tensor = self.pad_embedding
        # Iterate over the tensor in chunks of seq_len
        for i in range(0, tensor.size(0) - self.seq_len + 1, self.seq_len):
            input_tensor = tensor[i:i + self.seq_len]
            output_tensor = input_tensor[:-1]
            output_tensor = torch.cat((pad_tensor, output_tensor), dim=0)
            yield input_tensor, output_tensor


    def __len__(self):
        return math.ceil(len(self.tensors) / self.seq_len) # Number of pairs of tensors

    def clear_tensors(self):
        del self.tensors_loaded
        del self.tensors
        self.tensors = None
        self.tensors_loaded = None

loaded_data = torch.load("train/extra_tokens.pt")
pad_input_embedding = loaded_data['pad']

batch_sz = 256
learning_rate = 0.001

model = AutoModel.from_pretrained("t5-base")
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# NEEEED TO UPDATE
train_file_paths = ["train/output_0_30000.pt",
                    "train/output_30000_50000.pt",  
                    "train/output_50000_100000.pt", 
                    "train/output_100000_150000.pt",
                    "train/output_150000_200000.pt",
                    "train/output_200000_250000.pt",
                    "train/output_250000_300000.pt",
                    "train/output_300000_400000.pt",
                    "train/output_400000_500000.pt",
                    "train/output_500000_600000.pt",
                    "train/output_600000_700000.pt",
                    "train/output_700000_800000.pt",
                    "train/output_800000_900000.pt",
                    "train/output_900000_1000000.pt",
                    "train/output_1000000_1100000.pt",
                    "train/output_1100000_1200000.pt",
                    "train/output_1200000_1300000.pt",
                    "train/output_1300000_1400000.pt",
                    "train/output_1400000_1500000.pt",
                    "train/output_1500000_1600000.pt",
                    "train/output_1600000_1800000.pt",
                    "train/output_1800000_2000000.pt",
                    "train/output_2000000_2200000.pt",
                    "train/output_2200000_2400000.pt",
                    "train/output_2400000_2600000.pt",
                    "train/output_2600000_2800000.pt",
                    "train/output_2800000_3000000.pt",
                    "train/output_3000000_3100000.pt",
                    "train/output_3100000_3200000.pt",
                    "train/output_3200000_3300000.pt",
                    "train/output_3300000_3400000.pt",
                    "train/output_3400000_3500000.pt",
                    "train/output_3500000_3600000.pt",
                    "train/output_3600000_3700000.pt",
                    "train/output_3700000_3800000.pt",
                    "train/output_3800000_3900000.pt",
                    "train/output_3900000_4000000.pt",
                    "train/output_4000000_4100000.pt",
                    "train/output_4100000_4200000.pt",
                    "train/output_4200000_4400000.pt",
                    "train/output_4400000_4600000.pt",
                    "train/output_4600000_4800000.pt",
                    "train/output_4800000_4900000.pt",
                    "train/output_4900000_5000000.pt",
                    "train/output_5000000_5200000.pt",
                    "train/output_5200000_5400000.pt",
                    "train/output_5400000_5600000.pt",
                    "train/output_5600000_5800000.pt",
                    "train/output_5800000_6000000.pt",
                    "train/output_6000000_6150000.pt"]

test_file_paths = ["test/output_6150000_6300000.pt", 
                    "test/output_6300000_6500000.pt"]

seq_len = 32

file_name_template = "T5_base_epoch_{}.pth"
for epoch in range(0, 10):
    train_loader_size = 0
    test_loader_size = 0
    epoch_loss = 0
    file_num = 0
    for file_path in train_file_paths:
        train_dataset = FinalDataset(file_path, seq_len, pad_input_embedding)
        file_num += 1
        train_loader = DataLoader(train_dataset, batch_size=batch_sz)

        for inputs, targets in tqdm(train_loader, desc=f"Processing File {file_num} of train data"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            train_loader_size += targets.size(0)

            llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = targets)
            llmm_output = llmm_output[0]

            shift_targets = inputs[:, :, :]
            shift_labels = llmm_output[:, :, :]

            loss = criterion(shift_labels, shift_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_dataset.clear_tensors()

        if file_num % 15 == 0:
            checkpoint = {
                'epoch': epoch,
                'file_done': file_num,
                'epoch_loss': epoch_loss,
                'train_loader_size': train_loader_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            file_name = file_name_template.format(epoch)
            torch.save(checkpoint, file_name)


    valid_loss = 0
    file_num = 0
    for file_path in test_file_paths:
        test_dataset = FinalDataset(file_path, seq_len, pad_input_embedding)
        file_num += 1
        test_loader = DataLoader(test_dataset, batch_size=batch_sz)

        for inputs, targets in tqdm(test_loader, desc=f"Processing File {file_num} of test data"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            test_loader_size += targets.size(0)

            with torch.no_grad():

                llmm_output = model(inputs_embeds = inputs, decoder_inputs_embeds = targets)
                llmm_output = llmm_output[0]

                shift_targets = inputs[:, :, :]
                shift_labels = llmm_output[:, :, :]

                loss_func = criterion(shift_labels, shift_targets)

            loss = loss_func.item()
            valid_loss += loss

        test_dataset.clear_tensors()

    print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
    current_directory = os.getcwd()
    with open(os.path.join(current_directory, 'epoch_loss.txt'), 'a') as file:
        file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    file_name = file_name_template.format(epoch)
    torch.save(checkpoint, file_name)

