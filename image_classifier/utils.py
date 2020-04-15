import numpy as np
import torch
import json
from train_helper import build_model
from pathlib import Path

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a pytorch tensor
    '''
    from PIL import Image
    # resize to 256x256 and crop to 224x224
    pil_img = Image.open(image).resize((256, 256))
    w, h = pil_img.size
    pil_img = pil_img.crop(((w - 224)/2, (h - 224)/2, (w + 224)/2, (h + 224)/2 ))
    
    # normalize
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_img = np.array(pil_img) / 255
    np_img = (np_img - means) / stds
    
    # reordering dimensions
    np_img = np_img.transpose(2, 0, 1)
    pyt_img = torch.from_numpy(np_img).float()
    # adding batch dimension and returning
    return torch.unsqueeze(pyt_img, 0)

def load_mapping(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model = build_model(arch=checkpoint['arch'], 
                        hidden_units=checkpoint['hidden_units'])
    model.classifier.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    if not checkpoint['fully_trained']:
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['optimizer_lr'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer
    else:
        return model
    
def save_checkpoint(model, class_to_idx, hidden_units, lr, arch, save_dir,
                    e = 0, fully_trained = True, optimizer = None, filename=None):
    # create save_dir if it doesn't exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {'hidden_units': hidden_units,
                  'model_state': model.classifier.state_dict(),
                  'epoch': e,
                  'lr': lr,
                  'arch': arch,
                  'fully_trained': fully_trained,
                  'class_to_idx': class_to_idx,
                  'idx_to_class': {v:k for k, v in class_to_idx.items()}}
    if filename is None:
        filename = arch + '.pth'
    if not fully_trained:
        checkpoint['optimizer_state'] = optimizer.state_dict()
        checkpoint['optimizer_lr'] = optimizer.param_groups[0]['lr']
    torch.save(checkpoint, save_dir + '/' + filename)