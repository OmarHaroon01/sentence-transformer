### HUGGINGFACE AUTHORIZATION FOR ACCESSING LLAMA MODEL ###
from huggingface_hub import login
login(token = "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")

### Importing required Libraries ###
from transformers import AutoModel, AutoConfig, set_seed, get_scheduler
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, DataLoader
import math

### LIST FOR STORING TRAIN AND TEST FILES ###
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

### Setting up seed for reproducibility and device  ###
set_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FinalDataset(IterableDataset):
    """
    FinalDataset is an iterable dataset class designed to handle sequential data stored in tensor files.
    
    This class:
    - Loads tensor data from a specified file.
    - Concatenates the tensor data.
    - Iterates over the tensor data in chunks of a specified sequence length.
    - Provides the total number of such sequences in the dataset.
    - Offers a method to clear the tensor data from memory.
    
    Attributes:
        file_path (str): The path to the file containing the tensor data.
        tensors (Tensor): The tensor data loaded from the file and processed.
        seq_len (int): The length of the sequences to iterate over.
    """
    def __init__(self, file_path, seq_len):
        self.file_path = file_path
        self.tensors = torch.load(file_path, map_location=torch.device('cpu'))
        self.tensors = torch.cat(self.tensors, dim = 0)
        self.seq_len = seq_len

    def __iter__(self):
        tensor = self.tensors
        # Iterate over the tensor in chunks of seq_len
        for i in range(0, tensor.size(0) - self.seq_len + 1, self.seq_len):
            input_tensor = tensor[i:i + self.seq_len]
            yield input_tensor


    def __len__(self):
        return math.floor(len(self.tensors) / self.seq_len) # Number of pairs of tensors

    def clear_tensors(self):
        del self.tensors
        self.tensors = None

### Setting batch_sz and LR ###
batch_sz = 4
learning_rate = 0.001


class SharedProjection(nn.Module):
    """
    SharedProjection is a class used for sharing embeddings of projection.
    
    This class:
    - Initializes a weight matrix for projection.
    - Provides a forward method to perform either normal or transpose projection based on the input.
    
    Attributes:
        embedding_size (int): The size of the input embedding.
        d_model (int): The size of the model's output.
        weight (Tensor): The weight matrix used for projection.
    """
    def __init__(self, embedding_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, embedding_size, dtype=torch.float16))

    def forward(self, x, reverse=False):
        if reverse:
            # Perform the transpose projection
            return torch.matmul(x, self.weight)
        else:
            # Perform the normal projection
            return torch.matmul(x, self.weight.t())

class LlamaWithProjection(nn.Module):
    """
    LlamaWithProjection is a neural network module that integrates a pretrained LLaMA model with a shared projection layer.

    This class:
    - Initializes a shared projection layer for dimensional transformations.
    - Loads a pretrained LLaMA model, freezing most of its parameters except the last few layers.
    - Projects input embeddings before and after passing through the LLaMA model.

    Attributes:
        embedding_size (int): The size of the input embedding.
        d_model (int): The size of the model's internal representation.
        DEVICE (str): The device to run the model on (e.g., 'cpu', 'cuda').
        shared_projection (SharedProjection): The projection layer for input and output transformations.
        llama (AutoModel): The pretrained LLaMA model with some layers unfrozen.
    """
    def __init__(self, embedding_size, d_model, DEVICE):
        super().__init__()
        self.embedding_size = embedding_size
        self.d_model = d_model
        self.DEVICE = DEVICE

        self.shared_projection = SharedProjection(embedding_size, d_model)
        self.llama = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map=DEVICE)

        #Freezing all parameters of llama
        for param in self.llama.parameters():
            param.requires_grad = False

        
        #Unfreezing last few layers of llama
        for layer_index in range(28, 32):
            for param in self.llama.layers[layer_index].parameters():
                param.requires_grad = True

    def forward(self, x):
        x = x.to(torch.float16).to(self.DEVICE)
        x = self.shared_projection(x)
        x = self.llama(inputs_embeds = x.to(self.DEVICE))[0]
        x = self.shared_projection(x, reverse=True)
        return x

### Defining Model ###
model = LlamaWithProjection(1024, 4096, DEVICE)
model.to(DEVICE)

### Defining loss function, optimizer and LR scheduler ###
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
training_step = 44628480 // batch_sz
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=3000,
    num_training_steps=training_step
)

### seq_len was defined as 32 as paragraphs of wiki had an average of 18 sentence per paragraph ###
seq_len = 32


### Training & Validation Loop ###
file_name_template = "Llama_epoch_{}.pth"
for epoch in range(0, 10):
  model.train()
  train_loader_size = 0
  test_loader_size = 0
  epoch_loss = 0
  file_num = 0
  for file_path in train_file_paths:
    train_dataset = FinalDataset(file_path, seq_len)
    file_num += 1
    train_loader = DataLoader(train_dataset, batch_size=batch_sz)
    for inputs in tqdm(train_loader, desc=f"Processing File {file_num} of train data"):
      ### Shifting to float16 for matching Dtype as that of model
      inputs = inputs.to(torch.float16).to(DEVICE)

      train_loader_size += inputs.size(0)

      llmm_output = model(inputs)

      shift_targets = inputs[:, 1:, :]
      shift_labels = llmm_output[:, :-1, :]

      loss = criterion(shift_labels, shift_targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      epoch_loss += loss.item()

    train_dataset.clear_tensors()

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
    test_dataset = FinalDataset(file_path, seq_len)
    file_num += 1
    test_loader = DataLoader(test_dataset, batch_size=batch_sz)
    for inputs in tqdm(test_loader, desc=f"Processing File {file_num} of test data"):
      inputs = inputs.to(torch.float16).to(DEVICE)

      test_loader_size += inputs.size(0)

      with torch.no_grad():
        llmm_output = model(inputs)

        shift_targets = inputs[:, 1:, :]
        shift_labels = llmm_output[:, :-1, :]

        loss_func = criterion(shift_labels, shift_targets)

      loss = loss_func.item()
      valid_loss += loss
    test_dataset.clear_tensors()

  
  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss_llama_8B.txt'), 'a') as file:
      file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/(train_loader_size // batch_sz)}, Eval Loss: {valid_loss/(test_loader_size // batch_sz)}, Training Data Size: {train_loader_size}, Test Data Size: {test_loader_size}\n')
  checkpoint = {
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': lr_scheduler.state_dict()
  }
  file_name = file_name_template.format(epoch)
  torch.save(checkpoint, file_name)
