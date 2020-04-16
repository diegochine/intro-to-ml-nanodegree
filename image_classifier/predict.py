import argparse
import sys
import torch
from utils import load_mapping, load_checkpoint, process_image

if __name__  == "__main__":
    # setting command line arguments
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('img_path', action='store', type=str, help='Path to image (.jpg)')
    parser.add_argument('checkpoint', action='store', type=str, help='Path to checkpoint (.pth)')
    # Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    parser.add_argument('--top_k', action='store', default=1, type=int, help='Top k most likely classes')
    # Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument('--category_names', action='store', default='cat_to_name.json', type=str, help='JSON categories mapping')
    # Use GPU for inference: python predict.py input checkpoint --gpu
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for inference')
    options = parser.parse_args(sys.argv[1:])

    # choosing device (gpu or cpu)
    if options.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # loading categories mapping
    cat_to_name = load_mapping(options.category_names)
    # loading and setting up model
    model = load_checkpoint(options.checkpoint)
    model.eval()
    model.to(device)

    # loading image
    img_tensor = process_image(options.img_path).to(device)

    # compute output
    with torch.no_grad():
        output = model.forward(img_tensor)
        ps = torch.exp(output)
        probs, classes = ps.topk(options.top_k, dim=1)
    
    # print results
    probs = probs.tolist()[0]
    classes = [model.idx_to_class[c.item()] for c in classes[0]]
    results = {cat_to_name[c]: prob for c, prob in zip(classes, probs)}
    print('{:20} | {}'.format('CLASS', 'PROBABILITY'))
    print('-'*34)
    for c, p in results.items():
        print('{:20} | {:.2f}%'.format(c, p*100))