# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset
Usage - train:
    $ python classifier.py --model efficientnet_b0 --data path/to/dataset --project second_classifier --name 20220510_model2 --epochs 15 --img 224
"""

import argparse
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.cuda import amp
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import Classify, DetectMultiBackend
from utils.general import NUM_THREADS, download, check_file, increment_path, check_git_status, check_requirements, \
    colorstr
from utils.torch_utils import model_info, select_device, de_parallel

# Functions
normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std
denormalize = lambda x, mean=0.5, std=0.25: x * std + mean


def train():
    save_dir, data, bs, epochs, nw, imgsz = Path(opt.save_dir), opt.data, opt.batch_size, opt.epochs, \
                                            min(NUM_THREADS, opt.workers), opt.img_size

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Download Dataset
    data_dir = FILE.parents[1] / 'datasets' / data
    if not data_dir.is_dir():
        url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
        download(url, dir=data_dir.parent)

    # Transforms
    trainform = T.Compose([# T.RandomGrayscale(p=0.01),
                           T.RandomHorizontalFlip(p=0.5),
                           T.RandomVerticalFlip(p=0.5),
                           # T.RandomAffine(degrees=1, translate=(.2, .2), scale=(1 / 1.5, 1.5),
                           #                shear=(-1, 1, -1, 1), fill=(114, 114, 114)),
                           # T.Resize([imgsz, imgsz]),  # very slow
                           T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])  # PILImage from [0, 1] to [-1, 1]
    testform = T.Compose(trainform.transforms[-2:])

    # Dataloaders
    trainset = torchvision.datasets.ImageFolder(root=data_dir / 'train', transform=trainform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw)
    valset = torchvision.datasets.ImageFolder(root=data_dir / 'val', transform=testform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True, num_workers=nw)
    names = trainset.classes
    nc = len(names)
    print(f'Training {opt.model} on {data} dataset with {nc} classes...')
    
    # save class names in txt file
    with open(save_dir / 'class_names.txt', mode = 'w') as f:
        f.write(', '.join(names))

    # Show images
    images, labels = iter(trainloader).next()
    imshow(denormalize(images[:64]), labels[:64], names=names, f=save_dir / 'train_images.jpg')

    # Model
    if opt.model.startswith('yolov5'):
        # YOLOv5 Classifier
        model = torch.hub.load('ultralytics/yolov5', opt.model, pretrained=True, autoshape=False)
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:10] if opt.model.endswith('6') else model.model[:8]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else sum([x.in_channels for x in m.m])  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        for p in model.parameters():
            p.requires_grad = True  # for training
    elif opt.model in torch.hub.list('rwightman/gen-efficientnet-pytorch'):  # i.e. efficientnet_b0
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', opt.model, pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, nc)
    else:  # try torchvision
        model = torchvision.models.__dict__[opt.model](pretrained=True)
        model.fc = nn.Linear(model.fc.weight.shape[1], nc)

    # print(model)  # debug
    model_info(model)

    # Optimizer
    lr0 = 0.0001 * bs  # intial lr
    lrf = 0.01  # final lr (fraction of lr0)
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr0 / 10)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

    # Scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # Train
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    best_fitness = 0.0
    # scaler = amp.GradScaler(enabled=cuda)
    print(f'Image sizes {imgsz} train, {imgsz} val\n'
          f'Using {nw} dataloader workers\n'
          f"Logging results to {colorstr('bold', save_dir)}\n"
          f'Starting training for {epochs} epochs...\n\n'
          f"{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        mloss = 0.0  # mean loss
        model.train()
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = resize(images.to(device)), labels.to(device)

            # Forward
            with amp.autocast(enabled=False):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            loss.backward()  # scaler.scale(loss).backward()

            # Optimize
            optimizer.step()  # scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

            # Print
            mloss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}"

            # Validation
            if i == len(pbar) - 1:
                fitness = test(model, valloader, names, criterion, pbar=pbar)  # validation

        # Scheduler
        scheduler.step()

        # Best fitness
        if fitness > best_fitness:
            best_fitness = fitness

        # Save model
        final_epoch = epoch + 1 == epochs
        if (not opt.nosave) or final_epoch:
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'optimizer': None,  # optimizer.state_dict()
                    'date': datetime.now().isoformat()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fitness:
                torch.save(ckpt, best)
            del ckpt

    # Train complete
    if final_epoch:
        print(f'Training complete. Results saved to {save_dir}.')

        # # Show predictions
        # images, labels = iter(valloader).next()
        # images = resize(images.to(device))
        # pred = torch.max(model(images), 1)[1]
        # imshow(denormalize(images), labels, pred, names, verbose=True, f=save_dir / 'validation_images.jpg')

    return best, names  


def test(model, dataloader, names, criterion=None, verbose=False, pbar=None, conf_mat = False):
    model.eval()
    pred, targets, loss = [], [], 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = resize(images.to(device)), labels.to(device)
            y = model(images)
            pred.append(torch.max(y, 1)[1])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets == pred).float()

    if pbar:
        pbar.desc += f"{loss / len(dataloader):<12.3g}{correct.mean().item():<12.3g}"

    accuracy = correct.mean().item()
    if verbose:  # all classes
        # slight modification by KM for text formatting
        print(f"{'class':25}{'number':>10}{'accuracy':>10}")
        print(f"{'all':25}{correct.shape[0]:10}{accuracy:10.5g}")
        for i, c in enumerate(names):
            t = correct[targets == i]
            print(f"{c:25}{t.shape[0]:10}{t.mean().item():10.5g}")
        print()

    # todo: generate confusion matrix
    if conf_mat:
        pass

    return accuracy


def classify(model, size=128, file='../datasets/mnist/test/3/30.png', plot=False):
    # YOLOv5 classification model inference
    import cv2
    import numpy as np
    import torch.nn.functional as F

    resize = torch.nn.Upsample(size=(size, size), mode='bilinear', align_corners=False)  # image resize

    # Image
    im = cv2.imread(str(file))[..., ::-1]  # HWC, BGR to RGB
    im = np.ascontiguousarray(np.asarray(im).transpose((2, 0, 1)))  # HWC to CHW
    im = torch.tensor(im).float().unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    im = resize(normalize(im))

    # Inference
    results = model(im)
    p = F.softmax(results, dim=1)  # probabilities
    i = p.argmax()  # max index
    print(f'{file} prediction: {i} ({p[0, i]:.2f})')

    # Plot
    if plot:
        denormalize = lambda x, mean=0.5, std=0.25: x * std + mean
        imshow(denormalize(im), f=Path(file).name)

    return p


def imshow(img, labels=None, pred=None, names=None, nmax=64, verbose=False, f=Path('images.jpg')):
    # Show classification image grid with labels (optional) and predictions (optional)
    import matplotlib.pyplot as plt

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(img.cpu(), len(img), dim=0)  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n ** 0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m, tight_layout=True)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)))  # cmap='gray'
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s)

    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    print(colorstr('imshow: ') + f"examples saved to {f}")

    if verbose and labels is not None:
        print('True:     ', ' '.join(f'{names[i]:3s}' for i in labels))
    if verbose and pred is not None:
        print('Predicted:', ' '.join(f'{names[i]:3s}' for i in pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
    parser.add_argument('--data', type=str, default='mnist', help='cifar10, cifar100, mnist or mnist-fashion')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=128, help='train, test image sizes (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Checks
    check_git_status()
    check_requirements()

    # Parameters
    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != 'cpu'
    opt.hyp = check_file(opt.hyp)  # check files
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    resize = torch.nn.Upsample(size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)  # image resize

    # Train
    best, names = train()

    # Test (if available)
    test_dir = FILE.parents[1] / 'datasets' / opt.data / 'test'
    if os.path.isdir(test_dir):
        testform = T.Compose([T.ToTensor(),
                              T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])  # PILImage from [0, 1] to [-1, 1])
        testset = torchvision.datasets.ImageFolder(root=test_dir, transform=testform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=min(NUM_THREADS, opt.workers))

        print(f"test by {best}")
        best_model = torch.load(best, map_location=torch.device('cpu'))['model'].float()
        best_model.to("cuda:0")
        test(best_model, testloader, names, verbose=True)
        