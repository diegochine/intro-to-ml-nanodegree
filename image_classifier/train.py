import argparse
import sys
from train_helper import *
from utils import save_checkpoint
from torch import optim
from workspace_utils import active_session

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('data_dir', action='store', type=str)
    # Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    parser.add_argument('--save_dir', action='store', default='./checkpoints', type=str)
    # Choose architecture: python train.py data_dir --arch "vgg13"
    parser.add_argument('--arch', action='store', default='vgg13', type=str)
    # Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    parser.add_argument('--learning_rate', action='store', default='0.001', type=float)
    parser.add_argument('--hidden_units', action='append', default=[], type=int)
    parser.add_argument('--epochs', action='store', default='10', type=int)
    # Use GPU for training: python train.py data_dir --gpu
    parser.add_argument('--gpu', action='store_true', default=False)
    options = parser.parse_args(sys.argv[1:])
    if not options.hidden_units:
        options.hidden_units = [512]
    
    # choosing device (gpu or cpu)
    if options.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    # loading data
    trainloader, validloader, _, class_to_idx = load_data(options.data_dir)
    
    # building model and optimizer, and setting loss function
    model = build_model(arch=options.arch, hidden_units=options.hidden_units)
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=options.learning_rate)
    # training loop
    print('Training loop starting')
    with active_session():
        for e in range(options.epochs):
            train_loss = 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # validation loop
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    valid_loss += criterion(logps, labels).item()
                    ps = torch.exp(logps)
                    _, predictions = ps.topk(1, dim=1)
                    correct = predictions == labels.view(*predictions.shape)
                    accuracy += torch.mean(correct.type(torch.FloatTensor))

                print('EPOCH {:2}/{:2}: Train loss: {:.3f} - Valid loss: {:.3f} - Accuracy (on validation): {:.2f}%'.format(
                       e+1, options.epochs, train_loss/len(trainloader), valid_loss/len(validloader), accuracy*100/len(validloader)))
                model.train()
    save_checkpoint(model, class_to_idx, hidden_units=options.hidden_units, lr=options.learning_rate, arch=options.arch, save_dir=options.save_dir)