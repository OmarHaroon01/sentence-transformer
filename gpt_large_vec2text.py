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
batch_sz = 64
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
        self.weight = nn.Parameter(torch.randn(d_model, embedding_size))

    def forward(self, x, reverse=False):
        if reverse:
            # Perform the transpose projection
            return torch.matmul(x, self.weight)
        else:
            # Perform the normal projection
            return torch.matmul(x, self.weight.t())

class GPT2WithProjection(nn.Module):
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
        self.config = AutoConfig.from_pretrained("gpt2-large")
        self.gpt = AutoModel.from_config(self.config)

        for param in self.gpt.wte.parameters():
          param.requires_grad = False

    def forward(self, x):
        x = x.to(self.DEVICE)
        x = self.shared_projection(x)
        x = self.gpt(inputs_embeds = x.to(self.DEVICE))[0]
        x = self.shared_projection(x, reverse=True)
        return x

### Defining Model ###
model = GPT2WithProjection(768, 1280, DEVICE)
model.to(DEVICE)

### Defining loss function, optimizer and LR scheduler ###
criterion = nn.MSELoss()
no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
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
file_name_template = "gpt_large_vec2text_epoch_{}.pth"
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
      inputs = inputs.to(DEVICE)

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
      inputs = inputs.to(DEVICE)

      test_loader_size += inputs.size(0)

      with torch.no_grad():
        llmm_output = model(inputs)

        shift_targets = inputs[:, 1:, :]
        shift_labels = llmm_output[:, :-1, :]

        loss_func = criterion(shift_labels, shift_targets)

      loss = loss_func.item()
      valid_loss += loss
    test_dataset.clear_tensors()

  
  print(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/train_loader_size}, Eval Loss: {valid_loss/test_loader_size}')
  current_directory = os.getcwd()
  with open(os.path.join(current_directory, 'epoch_loss_gpt_large.txt'), 'a') as file:
      file.write(f'Epoch [{epoch+1}], Train Loss: {epoch_loss/train_loader_size}, Eval Loss: {valid_loss/test_loader_size}\n')
  checkpoint = {
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': lr_scheduler.state_dict()
  }
  file_name = file_name_template.format(epoch)
  torch.save(checkpoint, file_name)