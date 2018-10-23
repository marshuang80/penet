import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, X):
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        out = self.classifier(lstm_out.view(len(sentence), -1))
        out = F.sigmoid(out)
        return out

def train_lstm(input, hidden_dim):
    model = LSTMClassifier(len(input), hidden_dim, 1)
    criterion = nn.BCEWithLogitsLoss() # TODO: what loss should we use?
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []
    valid_losses = []
    epochs = []

    best_val_loss = float('Inf')

    for epoch in range(lstm_epochs):
        model.zero_grad()
        model.hidden = model.init_hidden()

        tag_scores = model(sentence_in)

        loss = criterion(tag_scores, targets)
        
        # Only if train
        loss.backward()
        optimizer.step()

        print("LSTM epoch start:", epoch)
        train_loss = 

        print(f"LSTM: Average training loss {train_loss:0.4f}")

        val_loss = 

        print(f"LSTM: Average validation loss {val_loss:0.4f}")
        
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        epochs.append(epoch)
