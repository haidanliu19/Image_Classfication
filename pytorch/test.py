from utils.general import (add_pr_curve_tensorboard)
import torch.nn.functional as F
import torch

from tqdm import tqdm

def test(test_loader, model, classes, criterion, training, rank, tb_writer):
    test_loss, correct = 0.0, 0
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    

    size = len(test_loader.dataset)    
    nb = len(test_loader)  
    pbar = enumerate(test_loader)
    if rank in (-1, 0):
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    # 모델 평가하기
    class_probs = []
    class_label = []
    with torch.no_grad():
        for n_iter, data_point in pbar:
            inputs, targets = data_point
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device)
            output = model(inputs)
            test_loss += criterion(output, targets).item()
            correct += (output.argmax(1) == targets).type(torch.float).sum().item()
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(targets)

            test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
            test_label = torch.cat(class_label)
    model.train()
    
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_label)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds, classes, tb_writer)
    
    test_loss /= nb
    correct /= size
    
    return test_loss, correct