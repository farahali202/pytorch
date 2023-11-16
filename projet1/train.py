# import dependencies
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class FeedForwardNet(nn.Module):

    # constructor
    def __init__(self):  
        super(FeedForwardNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),  
            nn.ReLU(),
            nn.Linear(256, 10)  
        )
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, input_data):  
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)  
        predictions = self.softmax(logits)
        return predictions

# download dataset
def download_mnist_data():
    
    train_data = datasets.MNIST(
        root="data",  
        download=True,
        train=True,
        transform=ToTensor()  
    )
    validation_data = datasets.MNIST(
        root="data",  
        download=True,
        train=False,
        transform=ToTensor()  
    )
    return train_data, validation_data

#train model
def train_one_epoch(model,data_loader,loss_fn,optimizer,device):
    
    for inputs,targets in data_loader:
        inputs,targets=inputs.to(device),targets.to(device)
        
       
        predictions=model(inputs)
        loss=loss_fn(predictions,targets)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss :{loss.item()}")

def train(model,data_loader,loss_fn,optimizer,device,epocks):
    for i in range(epocks):
        train_one_epoch(model,data_loader,loss_fn,optimizer,device)
        print('_________________________')
    print("training is done")



if __name__ == '__main__':

    train_data, validation_data = download_mnist_data()
    print('data download')

    BATCH_SIZE = 128
    epocks=10
    learning_rate=.001
    train_data_loader = DataLoader(train_data,
                                   batch_size=BATCH_SIZE
                                   )

    # build the model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'device: {device}')

    feed_forward_net = FeedForwardNet().to(device)  


    #instantiate loss fucntion/optimizer
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(feed_forward_net.parameters(),lr=learning_rate)
    
    #train model
    train(feed_forward_net,train_data_loader,loss_fn,optimizer,device,epocks)

    #sore model
    torch.save(feed_forward_net.state_dict(),'feedforward.pth')
    print("model trained and stored ")










