import torch
import torchvision
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

class stupidassmodel(torch.nn.Module):
    def __init__(self):
        super(stupidassmodel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.activation1 = torch.nn.ReLU6()
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.activation2 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv2d(16, 16, 5)
        self.activation3 = torch.nn.ReLU6()
        self.fc1 = torch.nn.Linear(16 * 20 * 20, 10)
        self.activation4 = torch.nn.ReLU6()
        
    def forward(self, x):
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        x = self.activation3(self.conv3(x))

        x = x.view(-1, 16 * 20 * 20)
        x = self.activation4(self.fc1(x))
        return x
    
    def my_train(self, epochs, the_bowl, train_loader):
        loss = 0
        for i in range(epochs):
            for input, label in tqdm(train_loader):
                input = input.cuda(0)
                label = label.cuda(0)
                shit = self.forward(input)

                loss += torch.nn.functional.cross_entropy(shit, label)
                loss.backward()

                the_bowl.step()
                the_bowl.zero_grad()
                loss = 0

    def evaluate(self, test_dataset):
        flush = 0
        for input, label in tqdm(test_dataset):
            input = input.cuda(0)
            testing_shit = self(input)

            THE_FINAL_SHIT = torch.argmax(testing_shit, dim=1) 
            THE_FINAL_SHIT = THE_FINAL_SHIT == label.cuda(0)
            for i in range(THE_FINAL_SHIT.shape[0]):
                if THE_FINAL_SHIT[i].item():
                    flush += 1

        return flush/10000
    
    def single_pred(self, test_dataset):
        for input, label in tqdm(test_dataset):
            print("input shape", input.shape)
            print("label shape", label.shape)
            input = input.cuda(0)
            testing_shit = self(input)
            print("testing_shit shape", testing_shit.shape)
            print("testing_shit dtype", testing_shit.dtype)

            THE_FINAL_SHIT = torch.argmax(testing_shit, dim=1) == label.cuda(0)
            print("THE_FINAL_SHIT shape", THE_FINAL_SHIT.shape)
            break
        return None
    
if __name__=='__main__':
    torch.backends.cudnn.enabled = False
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/',download=True, train=False, transform=ToTensor())

    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)

    batch_size_training=256
    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size_training, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)

    epochs = 5

    stupidass = stupidassmodel().cuda(0)
    toliet_func=torch.optim.SGD
    the_bowl = toliet_func(stupidass.parameters(), lr=0.001)
    loss = 0

    stupidass.my_train(epochs, the_bowl, train_loader)
    # stupidass = torch.load('stupidmodel.pt')

    print(stupidass.evaluate(test_loader))
    # stupidass.single_pred(test_loader)

    torch.save(stupidass.state_dict(), 'stupidmodel.pt')