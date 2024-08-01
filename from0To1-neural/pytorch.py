import torch
import torch.nn as nn
import  torch.optim as optim
import matplotlib.pyplot as plt
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

learning_rate = 0.1
epochs = 1000
losses = []
data = torch.tensor([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Dan
    [-27, -6],  # Emma
], dtype=torch.float32)
all_y_trues = torch.tensor([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Dan
    1,  # Emma
], dtype=torch.float32)



model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def train(data, all_y_trues):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, all_y_trues)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            losses.append(loss.item())
            # print(f"Epoch {epoch}, Loss: {loss.item()}")

    plt.plot(range(0, epochs, 10), losses)
    plt.title("loss over epochs")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), 'model.pth')


train(data, all_y_trues)

# 预测
# model.eval()
# with torch.no_grad():
#     predictions = model(data).detach().numpy()
# print("Predictions:", predictions)