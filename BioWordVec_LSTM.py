from torch import nn
from torch.nn import functional as F

embed_len = 200
hidden_dim = 100
n_layers=3

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        V = len(weights.key_to_index) + 1
        D = 300
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=weights.key_to_index['pad'])
        #self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        #self.embedding_layer = nn.Embedding(V, D)
        #self.embedding_layer.weight.data.copy_(embedding_weights)
        self.lstm = nn.LSTM(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, nb_classes)  ## Input dimension are 2 times hidden dimensions due to bidirectional results

    def forward(self, x):
        x = x.to(device)
        embeddings = self.embedding(x)
        embeddings= embeddings.to(device_model)
        hidden, carry = torch.randn(2*n_layers, len(x), hidden_dim), torch.randn(2*n_layers, len(x), hidden_dim)
        hidden , carry = hidden.to(device_model) , carry.to(device_model)
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        return self.linear(output[:,-1])