import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from collections import OrderedDict

def load_data(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(means, stds)]),
                       'valid': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(means, stds)]),
                       'test': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(means, stds)])
                    }
    
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data.class_to_idx

def build_model(arch='vgg19', hidden_units=[1024], p_drop=0.3, lr=0.01):
    t_models = {'vgg19': models.vgg19(pretrained=True),
                'densenet121': models.densenet121(pretrained=True),
                'resnet101': models.resnet101(pretrained=True)}
    
    # get pretrained model and freeze parameters
    model = t_models.get(arch, 'vgg19')
    for param in model.parameters():
        param.requires_grad = False
       
    # generate final layer and optimizer, based on the pretrained model
    if arch == 'vgg19':
        layers = [('input', nn.Linear(25088, hidden_units[0])), ('relu', nn.ReLU()), ('drop', nn.Dropout(p_drop))]
        for i, (s1, s2) in enumerate(zip(hidden_units[:-1], hidden_units[1:])):
            layers.append(('hl_' + str(i), nn.Linear(s1, s2)))
            layers.append(('relu_' + str(i), nn.ReLU()))
            layers.append(('drop_' + str(i), nn.Dropout(p_drop)))
        layers += [('out', nn.Linear(hidden_units[-1], 102)), ('logits', nn.LogSoftmax(dim=1))]
        # change last layer of pretrained model
        model.classifier = nn.Sequential(OrderedDict(layers))
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif arch == 'densenet121':
        layers = [('input', nn.Linear(1024, hidden_units[0])), ('relu', nn.ReLU()), ('drop', nn.Dropout(p_drop))]
        for i, (s1, s2) in enumerate(zip(hidden_units[:-1], hidden_units[1:])):
            layers.append(('hl_' + str(i), nn.Linear(s1, s2)))
            layers.append(('relu_' + str(i), nn.ReLU()))
            layers.append(('drop_' + str(i), nn.Dropout(p_drop)))
        layers += [('out', nn.Linear(hidden_units[-1], 102)), ('logits', nn.LogSoftmax(dim=1))]
        # change last layer of pretrained model
        model.classifier = nn.Sequential(OrderedDict(layers))
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif arch == 'resnet101':
        layers = [('input', nn.Linear(2048, hidden_units[0])), ('relu', nn.ReLU()), ('drop', nn.Dropout(p_drop))]
        for i, (s1, s2) in enumerate(zip(hidden_units[:-1], hidden_units[1:])):
            layers.append(('hl_' + str(i), nn.Linear(s1, s2)))
            layers.append(('relu_' + str(i), nn.ReLU()))
            layers.append(('drop_' + str(i), nn.Dropout(p_drop)))
        layers += [('out', nn.Linear(hidden_units[-1], 102)), ('logits', nn.LogSoftmax(dim=1))]
        # change last layer of pretrained model
        model.fc = nn.Sequential(OrderedDict(layers))
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        raise ValueError('Invalid/not supported architecture')
    return model, optimizer