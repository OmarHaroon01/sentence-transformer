from transformers import AutoModel, AutoConfig, get_scheduler
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader
import math
from datetime import datetime

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FinalDataset(IterableDataset):
    def __init__(self, file_path, seq_len, pad_embedding):
        self.file_path = file_path
        self.tensors = torch.load(file_path, map_location=torch.device('cpu'))
        self.tensors = torch.cat(self.tensors, dim = 0)
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
        del self.tensors
        self.tensors = None

loaded_data = torch.load("sonar/sonar_extra_tokens.pt")
pad_input_embedding = loaded_data['pad']

batch_sz = 256
learning_rate = 5e-5

config = AutoConfig.from_pretrained("gpt2-medium")
model = AutoModel.from_config(config)
model.to(DEVICE)

for param in model.wte.parameters():
    param.requires_grad = False

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
training_step = 44628480 // batch_sz
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=6000,
    num_training_steps= training_step 
)

train_file_paths = [
    "sonar/sonar_0_20000.pt", 
    "sonar/sonar_20000_50000.pt", 
    "sonar/sonar_50000_90000.pt", 
    "sonar/sonar_90000_120000.pt", 
    "sonar/sonar_120000_150000.pt", 
    "sonar/sonar_150000_200000.pt", 
    "sonar/sonar_200000_250000.pt", 
    "sonar/sonar_250000_300000.pt", 
    "sonar/sonar_300000_350000.pt", 
    "sonar/sonar_350000_400000.pt", 
    "sonar/sonar_400000_450000.pt",
    "sonar/sonar_450000_500000.pt", 
    "sonar/sonar_500000_550000.pt", 
    "sonar/sonar_550000_600000.pt", 
    "sonar/sonar_600000_650000.pt", 
    "sonar/sonar_650000_700000.pt", 
    "sonar/sonar_700000_750000.pt", 
    "sonar/sonar_750000_800000.pt", 
    "sonar/sonar_800000_850000.pt", 
    "sonar/sonar_850000_900000.pt", 
    "sonar/sonar_900000_950000.pt", 
    "sonar/sonar_950000_1000000.pt", 
    "sonar/sonar_1000000_1100000.pt", 
    "sonar/sonar_1100000_1200000.pt", 
    "sonar/sonar_1200000_1300000.pt", 
    "sonar/sonar_1300000_1400000.pt", 
    "sonar/sonar_1400000_1500000.pt", 
    "sonar/sonar_1500000_1600000.pt", 
    "sonar/sonar_1600000_1700000.pt", 
    "sonar/sonar_1700000_1800000.pt", 
    "sonar/sonar_1800000_1900000.pt", 
    "sonar/sonar_1900000_2000000.pt", 
    "sonar/sonar_2000000_2100000.pt", 
    "sonar/sonar_2100000_2200000.pt", 
    "sonar/sonar_2200000_2300000.pt", 
    "sonar/sonar_2300000_2400000.pt", 
    "sonar/sonar_2400000_2500000.pt", 
    "sonar/sonar_2500000_2600000.pt", 
    "sonar/sonar_2600000_2700000.pt", 
    "sonar/sonar_2700000_2800000.pt", 
    "sonar/sonar_2800000_2900000.pt", 
    "sonar/sonar_2900000_3000000.pt", 
    "sonar/sonar_3000000_3100000.pt", 
    "sonar/sonar_3100000_3200000.pt", 
    "sonar/sonar_3200000_3300000.pt", 
    "sonar/sonar_3300000_3400000.pt", 
    "sonar/sonar_3400000_3500000.pt", 
    "sonar/sonar_3500000_3600000.pt", 
    "sonar/sonar_3600000_3700000.pt", 
    "sonar/sonar_3700000_3800000.pt", 
    "sonar/sonar_3800000_3900000.pt", 
    "sonar/sonar_3900000_4000000.pt", 
    "sonar/sonar_4000000_4100000.pt", 
    "sonar/sonar_4100000_4200000.pt", 
    "sonar/sonar_4200000_4300000.pt", 
    "sonar/sonar_4300000_4400000.pt", 
    "sonar/sonar_4400000_4500000.pt", 
    "sonar/sonar_4500000_4600000.pt", 
    "sonar/sonar_4600000_4700000.pt", 
    "sonar/sonar_4700000_4800000.pt", 
    "sonar/sonar_4800000_4900000.pt", 
    "sonar/sonar_4900000_5000000.pt", 
    "sonar/sonar_5000000_5100000.pt", 
    "sonar/sonar_5100000_5200000.pt", 
    "sonar/sonar_5200000_5300000.pt", 
    "sonar/sonar_5300000_5400000.pt", 
    "sonar/sonar_5400000_5500000.pt", 
    "sonar/sonar_5500000_5600000.pt", 
    "sonar/sonar_5600000_5700000.pt", 
    "sonar/sonar_5700000_5800000.pt", 
    "sonar/sonar_5800000_5900000.pt", 
    "sonar/sonar_5900000_6000000.pt", 
    "sonar/sonar_6000000_6150000.pt"]


test_file_paths = [
    "sonar_test/sonar_6150000_6300000.pt", 
    "sonar_test/sonar_6300000_6500000.pt"]
    



seq_len = 32

print("TRAINING STARTED")
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
file_name_template = "gpt_medium_epoch_{}.pth"
for epoch in range(0, 10):
    model.train()
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

            llmm_output = model(inputs_embeds = inputs)
            llmm_output = llmm_output[0]

            shift_targets = inputs[:, 1:, :]
            shift_labels = llmm_output[:, :-1, :]

            loss = criterion(shift_labels, shift_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

        train_dataset.clear_tensors()
        print(f"File {file_num} of Epoch {epoch} completed")

        if file_num % 15 == 0:
            checkpoint = {
                'epoch': epoch,
                'file_done': file_num,
                'epoch_loss': epoch_loss,
                'train_loader_size': train_loader_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
            file_name = file_name_template.format(epoch)
            torch.save(checkpoint, file_name)

    model.eval()
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

                llmm_output = model(inputs_embeds = inputs)
                llmm_output = llmm_output[0]

                shift_targets = inputs[:, 1:, :]
                shift_labels = llmm_output[:, :-1, :]

                loss_func = criterion(shift_labels, shift_targets)

            loss = loss_func.item()
            valid_loss += loss

        test_dataset.clear_tensors()
        print(f"File {file_num} of Epoch {epoch} completed")

    print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/train_loader_size}, Eval Loss: {valid_loss/test_loader_size}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
    current_directory = os.getcwd()
    with open(os.path.join(current_directory, 'epoch_loss_gpt_medium.txt'), 'a') as file:
        file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/train_loader_size}, Eval Loss: {valid_loss/test_loader_size}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    file_name = file_name_template.format(epoch)
    torch.save(checkpoint, file_name)
print("TRAINING ENDED")
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
