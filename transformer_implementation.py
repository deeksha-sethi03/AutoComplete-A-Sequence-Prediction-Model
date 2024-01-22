from urllib.request import urlopen
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, random_split
from numpy import savetxt
from torch.nn import TransformerEncoder, TransformerEncoderLayer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################################

def ReadRawData(target_url):
  print('Reading Raw Data')
  raw_data = ''
  vocab = []
  for i in target_url:
    data = urlopen(i)
    for line in data:
      raw_data += ' ' + line.decode('utf-8-sig').strip()
      vocab.extend(line.decode('utf-8-sig').strip())
    print('Source Read: ',i)
  vocab = np.unique(np.array(vocab))
  vocabulary = {}
  for idx, x in enumerate(vocab):
    vocabulary[x] = idx
  return raw_data, vocabulary


target_url = ['https://www.gutenberg.org/cache/epub/100/pg100.txt', 'https://www.gutenberg.org/ebooks/2600.txt.utf-8', 'https://www.gutenberg.org/cache/epub/71999/pg71999.txt']
raw_data, vocabulary = ReadRawData(target_url)

##########################################################################

def DataEmbedding(dataset, vocabulary, n):
# print('Embedding the Dataset into One-Hot Enconding')
  embedded = np.zeros((n,len(vocabulary))).tolist()
  data_embedded = []
  labels = []
  for id, i in enumerate(dataset):
    embedded = np.zeros((n,len(vocabulary))).tolist()
    for idx, j in enumerate(i):
      embedded[idx][vocabulary[j]] = 1
    data_embedded.append(embedded)
  return data_embedded

##########################################################################

def DatasetCreation(raw_data, n):
  print('Creating the Dataset')

  sequence_length = n

  sequences = []
  labels = []

  for i in range(0, len(raw_data) - sequence_length):
      sequence = raw_data[i:i + sequence_length]
      label = raw_data[i + 1:i + sequence_length + 1]
      sequences.append(sequence)
      labels.append(label)



  dataset = np.column_stack((sequences, labels))
  return dataset

dataset = DatasetCreation(raw_data, n=32)
print('Sample Dataset: ', dataset[0])
dataset = dataset[:500000]

##########################################################################

def TrainTestSplit(dataset):
  print('Splitting the Dataset into Train & Test')
  # Define the split ratio (e.g., 80% train, 20% test)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size

  # Split the dataset into train and test subsets
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  return train_dataset, test_dataset

train_dataset, test_dataset = TrainTestSplit(dataset)
train_dataset = np.array(train_dataset)
print('Train Data Shape = ', train_dataset.shape)
test_dataset = np.array(test_dataset)
print('Test Data Shape = ', test_dataset.shape)

###########################################################################

class CustomDataLoader:
    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = len(data) // batch_size
        self.current_batch = 0

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start = self.current_batch * self.batch_size
        end = (self.current_batch + 1) * self.batch_size

        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]

        self.current_batch += 1

        return batch_data, batch_labels


# Define batch size and create a CustomDataLoader
print('Converting the Dataset into DataLoaders')

batch_size = 8
train_loader = CustomDataLoader(train_dataset[:, 0], train_dataset[:, -1], batch_size, shuffle=True)
test_loader = CustomDataLoader(test_dataset[:, 0], test_dataset[:, -1], batch_size, shuffle=True)

print('Vocabulary length = ',len(vocabulary))

###########################################################################

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.d_model = len(vocabulary)
        self.nhead = 14
        self.output_size = len(vocabulary)
        self.nlayers = 6
        encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        # self.Transformer = nn.Transformer(d_model = self.d_model, )
        self.linear = nn.Linear(self.d_model, self.output_size)

    def forward(self, src, src_mask):

        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        output = F.relu(output).reshape(-1, self.output_size)

        return output



#############################################################################

def train(net, optimizer, criterion, train_loader, test_loader, vocabulary, sequence_length, epochs, model_name, plot):
    model = net.to(device)
    # total_step = len(train_loader)
    overall_step = 0
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    output_size = len(vocabulary)

    for epoch in range(epochs):

        correct = 0
        total = 0
        running_loss = 0.0

        for i, (sequences, labels) in enumerate(train_loader):
            sequences = torch.Tensor(DataEmbedding(sequences, vocabulary, sequence_length))
            labels_raw = torch.Tensor(DataEmbedding(labels, vocabulary, sequence_length))
            labels = labels_raw.reshape(-1, output_size)
            labels = torch.argmax(labels, dim = 1)

            # Move tensors to configured device
            sequences = sequences.to(device)
            labels = labels.to(device)
            labels_raw = labels_raw.to(device)
            #Forward Pass
            outputs = model(sequences, None)
            # print('Ground-Truth Label = {} and Predicted Label = {}'.format(labels.shape, outputs.shape))
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if (i+1) % 1000 == 0:
              print ('Epoch [{}/{}], Iter: {}, Loss: {:.4f}, Accuracy [{}%]'.format(epoch+1, epochs, i, loss.item(), 100*correct/total))
            if plot:
              info = { ('loss_' + model_name): loss.item() }
            train_loss_values.append(loss.item())
            train_error.append(100-100*correct/total)
            # np.save('/home/deeksha/Documents/Assignments/Principles of Deep Learning/train_loss_values_transformer.npy', train_loss_values)
            # np.save('/home/deeksha/Documents/Assignments/Principles of Deep Learning/train_error_transformer.npy', train_error)


        # scheduler.step()
        print(f"Epoch {epoch + 1}: Learning Rate = {optimizer.param_groups[0]['lr']}")


        # print(f"Epoch {epoch + 1}: Validation Error = {100-100*correct/total}")


        val_running_loss = 0

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for i, (sequences, labels) in enumerate(test_loader):

                sequences = torch.Tensor(DataEmbedding(sequences, vocabulary, sequence_length))
                labels_raw = torch.Tensor(DataEmbedding(labels, vocabulary, sequence_length))
                labels = labels_raw.reshape(-1, output_size)
                labels = torch.argmax(labels, dim = 1)
                sequences = sequences.to(device)
                labels = labels.to(device)
                labels_raw = labels_raw.to(device)
                outputs = model(sequences,None)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_error.append(100-100*correct/total)
                val_loss_values.append(loss.item())
                # np.save('/home/deeksha/Documents/Assignments/Principles of Deep Learning/val_loss_values_transformer.npy', val_loss_values)
                # np.save('/home/deeksha/Documents/Assignments/Principles of Deep Learning/val_error_transformer.npy', val_error)


        print('Accuracy of the network on the test sequences: {} %'.format(100 * correct / total))


        # save to csv file
    return val_error,val_loss_values,train_error,train_loss_values

#######################################################################

model = TransformerNet().to(device)

epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
val_error,val_loss_values,train_error,train_loss_values= train(model, optimizer, criterion, train_loader, test_loader, vocabulary, 32, epochs, '_curve', True)
torch.save(model.state_dict(), 'trained_model.pth')
########################################################################

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label='Train Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_error, label='Train Error')
plt.plot(val_error, label='Validation Error')
plt.xlabel('Number of Weight Updates')
plt.ylabel('Error (%)')
plt.legend()

plt.show()


##############################################################

model = TransformerNet().to(device)
model.load_state_dict(torch.load('trained_model_transformer.pth'))


input = 'T'
sequence = '' + input
model.eval()
sequence_length = 1024
for i in range(sequence_length):
   encoded_input = torch.Tensor(DataEmbedding([input], vocabulary, 1)).reshape(1,1,-1)
   encoded_input = encoded_input.to(device)
   outputs = model(encoded_input)
   predicted_character = torch.argmax(outputs, dim = 1)
   for idx, i in enumerate(predicted_character):
      if(i == 1):
         input = vocabulary[idx]
   sequence += input

print(sequence)

######################################################################


