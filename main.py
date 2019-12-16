import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
import os 
import sys
import argparse


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs):
    res_loss = np.zeros((num_epochs,2))
    res_acc = np.zeros((num_epochs,2))
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for idx,phase in enumerate(['train', 'val']):
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm(model.parameters(),clip)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            res_loss[epoch,idx] = epoch_loss
            res_acc[epoch,idx] = epoch_acc
    return res_loss, res_acc

if __name__ == '__main__':
    
    clip = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",help="input model",type=str,choices=['alexnet','vgg16','resnet18'])
    parser.add_argument("--noise",help="input noise",type=str,choices=['true_label','random_label','random_pixel'])
    parser.add_argument("--num_epochs",help="number of epochs",type=int)
    parser.add_argument("--gpu",help="index of GPU",type=str)
    parser.add_argument("--no_regu",default=False,help="no regulalization",action='store_true')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    if args.no_regu:
        f_name_regu = 'noRegu'
    else:
        f_name_regu = 'Regu'
        
    print('save to acc/loss_{}_{}_{}'.format(args.model,args.noise,f_name_regu))
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    grey2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    transform_pretrain = transforms.Compose([transforms.Resize((224, 224),interpolation=2),
                                             transforms.ToTensor(),
                                             grey2rgb,
                                             normalize
                                             ])

    if args.noise == 'true_label':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_pretrain)
        batch_size = 20
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_pretrain)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                                  shuffle=False, num_workers=2)
        dataloaders = {'train':trainloader, 'val':testloader}
        dataset_sizes = {'train':len(trainset), 'val':len(testset)}
        
    elif args.noise == 'random_label':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_pretrain, target_transform=lambda y: torch.randint(0, 10, (1,)).item())
        batch_size = 20
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_pretrain)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                                  shuffle=False, num_workers=2)
        dataloaders = {'train':trainloader, 'val':testloader}
        dataset_sizes = {'train':len(trainset), 'val':len(testset)}
        
    elif args.noise == 'random_pixel':
        transform_pretrain = transforms.Compose([transforms.Resize((224, 224),interpolation=2),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x:torch.rand(x.size())),
                                         grey2rgb,
                                         normalize
                                         ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_pretrain)
        batch_size = 20
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_pretrain)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                                  shuffle=False, num_workers=2)
        dataloaders = {'train':trainloader, 'val':testloader}
        dataset_sizes = {'train':len(trainset), 'val':len(testset)}
        
    else:
        print('Error input noise')
        sys.exit()
        
    if args.model == 'alexnet':
        alexnet = models.alexnet(pretrained=False)
        num_ftrs = alexnet.classifier[6].in_features
        alexnet.classifier[6] = nn.Linear(num_ftrs, 10)
        criterion = nn.CrossEntropyLoss()
        if args.no_regu:
            alexnet.classifier[0].p = 0
            alexnet.classifier[3].p = 0
            alexnet = alexnet.to(device)
            optimizer_ft = optim.SGD(alexnet.parameters(), lr=0.02, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(alexnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
        else:
            alexnet = alexnet.to(device)
            optimizer_ft = optim.SGD(alexnet.parameters(), lr=0.02, momentum=0.9,weight_decay=0.00001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(alexnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
            
    elif args.model == 'vgg16':
        vgg16 = models.vgg16(pretrained=False)
        num_ftrs = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_ftrs, 10)
        criterion = nn.CrossEntropyLoss()
        if args.no_regu:
            vgg16.classifier[2].p = 0
            vgg16.classifier[5].p = 0
            vgg16 = vgg16.to(device)
            optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.1, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
        else:
            vgg16 = vgg16.to(device)
            optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.1, momentum=0.9,weight_decay=0.00001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
            
    elif args.model == 'resnet18':
        resnet18 = models.resnet18(pretrained=False)
        num_ftrs = resnet18.fc.in_features
        criterion = nn.CrossEntropyLoss()
        if args.no_regu:
            resnet18.fc = nn.Linear(num_ftrs, 10)
            resnet18 = resnet18.to(device)
            optimizer_ft = optim.SGD(resnet18.parameters(), lr=0.1, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(resnet18, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
        else:
            resnet18.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(num_ftrs, 10))
            resnet18 = resnet18.to(device)
            optimizer_ft = optim.SGD(resnet18.parameters(), lr=0.1, momentum=0.9,weight_decay=0.00001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.95)
            loss,acc = train_model(resnet18, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, args.num_epochs)
    
    else:
        print('Error input model')
        sys.exit()
    
    #print(acc, loss)
    np.save('acc_{}_{}_{}'.format(args.model,args.noise,f_name_regu),acc)
    np.save('loss_{}_{}_{}'.format(args.model,args.noise,f_name_regu),loss)
            
            
            
        