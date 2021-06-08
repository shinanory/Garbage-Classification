#导入库
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pylab
from torch.utils.data.dataloader import DataLoader

#路径
path = 'Garbage classification'
classes = os.listdir(path)

#验证图像
def showPic(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))
    pylab.show()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch): #训练
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch): #验证
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result): #输出
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


#获取gpu或cpu
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
#移动
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

#拟合模型
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    prob, preds  = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]


if __name__=='__main__':
    #处理图片的转换
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    #导入预置数据集ImageFolder  指定路径与转换操作
    dataset = ImageFolder(path, transform = transformations)

    #验证图像
    #img, label = dataset[89]
    #showPic(img, label)

    #设置随机数并随机拆分训练、验证、测试数据集
    rsd = 42
    torch.manual_seed(rsd)
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])

    #构建训练与验证的DataLoader
    train_dl = DataLoader(train_ds, 32, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_ds, 64, num_workers = 4, pin_memory = True)
    
    model = ResNet()

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    model = to_device(ResNet(), device)

    #训练拟合
    evaluate(model, val_dl)
    num_epochs = 2
    opt_func = torch.optim.Adam
    lr = 5.5e-5
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    #用一张图测试
    img, label = test_ds[51]
    plt.imshow(img.permute(1, 2, 0))
    pylab.show()
    print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))