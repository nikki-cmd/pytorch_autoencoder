import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


tensor_transform = transforms.ToTensor()

dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download=True,
                         transform = tensor_transform)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size = 32,
                                     shuffle=True)

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
            
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28*28)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return decoder


model = AE()

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
epochs = 20
outputs = []
losses = []

for epoch in range(epochs):
    for (image, _) in loader:
        image = image.reshape(-1, 28*28)
        
        reconstructed = model(image)
        
        loss = loss_function(reconstructed, image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
        print(f'epoch:{epoch}, loss:{loss}')
    outputs.append((epochs, image, reconstructed))

torch.save(model.state_dict(), 'models')


model = AE()
model.load_state_dict(torch.load('models', weights_only=True))
model.eval()

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses[-100:])

for i, item in enumerate(image):
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
 
for i, item in enumerate(reconstructed):
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])