from torch import nn
from torch.nn import functional as F
import torch
embed_len = 200
hidden_dim = 600
n_layers=1

nb_classes = 4
#device = 'cuda'
max_words = 250
k1 =max_words+1-3
k2 = max_words+1-4
k3 = max_words+1-5
class LSTM_CNN_merge(nn.Module):
    def __init__(self):
        super(LSTM_CNN_merge, self).__init__()
        V = len(weights.key_to_index) + 1
        D = 300
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=weights.key_to_index['pad'])
        #self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        #self.embedding_layer = nn.Embedding(V, D)
        #self.embedding_layer.weight.data.copy_(embedding_weights)
        self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True,
                            bidirectional=True , dropout=0.5)
        self.linear = nn.Linear(2*hidden_dim, 100)  ## Input dimension are 2 times hidden dimensions due to bidirectional results
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(3,2*hidden_dim), stride=1,padding=0),  # h = 9-3 +1  and w = 1 output : 7x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k1,1), stride=1)) #1x1
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(4,2*hidden_dim), stride=1,padding=0), #6x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k2,1), stride=1))  #1x1
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(5,2*hidden_dim), stride=1,padding=0), #5x1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(k3,1), stride=1)) #1X1
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1 * 1 * 100 * 3, 100)
        self.fc2 = nn.Linear(100, nb_classes)

    def forward(self, x):
        x = x.to(device)
        embeddings = self.embedding(x)
        embeddings = embeddings.to(device_model)
        hidden, carry = torch.randn(2*n_layers, len(x), hidden_dim), torch.randn(2*n_layers, len(x), hidden_dim)
        hidden , carry = hidden.to(device_model) , carry.to(device_model)
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        out1 = self.linear(output[:,-1])
        #print('output',output.shape)
        output = output.unsqueeze(1)
        #print('cnn input',output.shape)
        x1 = self.cnn1(output)
        x2 = self.cnn2(output)
        x3 = self.cnn3(output)
        x1 = self.drop_out(x1)
        x2 = self.drop_out(x2)
        x3 = self.drop_out(x3)
        #print('x',x1.shape)
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        x3 = x3.reshape(x3.size(0), -1)
        #print('x',x1.shape)
        out = torch.cat((x1,x2,x3),1)
        #print('out',out.shape)
        out2 = self.fc1(out)
        #print(out.shape)
        out = out1.add(out2)
        out = self.fc2(out)
        #print(out.shape)
        out = F.softmax(out, dim=1)
       
        #print(out.shape)
        return out
    