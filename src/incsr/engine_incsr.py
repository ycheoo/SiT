import math
import sys
from typing import Iterable

import numpy as np
import torch
import utils.misc as misc
from timm.utils import accuracy


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    epoch: int,
    domain_classnum: int,
):
    model.train(True)
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    output_list = []
    target_list = []
    optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets % domain_classnum

        with torch.cuda.amp.autocast():
            outputs = model(samples)["logits"]
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        output_list.append(torch.max(outputs, dim=1)[1])
        target_list.append(targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    scheduler.step()

    output_list = misc.all_concatenate(torch.cat(output_list, dim=0))
    target_list = misc.all_concatenate(torch.cat(target_list, dim=0))
    output_list, target_list = output_list.cpu().numpy(), target_list.cpu().numpy()

    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        output_list,
        target_list,
    )


@torch.no_grad()
def evaluate(model, data_loader, device, domain_classnum):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    print_freq = 100
    # switch to evaluation mode
    model.eval()

    output_list = []
    target_list = []
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples = batch[0]
        targets = batch[-1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets % domain_classnum

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(samples)["logits"]
            loss = criterion(outputs, targets)

        output_list.append(torch.max(outputs, dim=1)[1])
        target_list.append(targets)

        batch_size = samples.shape[0]
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    output_list = misc.all_concatenate(torch.cat(output_list, dim=0))
    target_list = misc.all_concatenate(torch.cat(target_list, dim=0))
    output_list, target_list = output_list.cpu().numpy(), target_list.cpu().numpy()

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}\n".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        output_list,
        target_list,
    )


@torch.no_grad()
def extract_features(model, data_loader, device, domain_classnum):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Extract features:"
    print_freq = 100
    # switch to evaluation mode
    model.eval()

    embed_list = []
    target_list = []

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples = batch[0]
        targets = batch[-1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = targets % domain_classnum

        # compute output
        with torch.cuda.amp.autocast():
            features = model(samples)["features"]

        embed_list.append(features)
        target_list.append(targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    embed_list = misc.all_concatenate(torch.cat(embed_list, dim=0))
    target_list = misc.all_concatenate(torch.cat(target_list, dim=0))

    print()

    return embed_list, target_list
