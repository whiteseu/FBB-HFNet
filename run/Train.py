import os
import torch
import tqdm
import sys

import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torch.distributed as dist

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.dataloader import *
from lib import *

@torch.no_grad()
def validation_mse(model, loader):
    """Pixel-mean MSE of final sigmoid predictions on the validation split."""
    module = model.module if hasattr(model, 'module') else model
    device = next(module.parameters()).device
    was_training = module.training
    module.eval()
    squared_error = 0.0
    pixel_count = 0
    try:
        for sample in loader:
            sample = {key: value.to(device, non_blocking=True)
                      if torch.is_tensor(value) else value
                      for key, value in sample.items()}
            logits = module(sample)['pred']
            target = sample['gt'].float()
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:],
                                       mode='bilinear', align_corners=True)
            if logits.shape != target.shape:
                raise ValueError('Validation prediction and mask shapes differ.')
            squared_error += F.mse_loss(
                logits.sigmoid().double(), target.double(),
                reduction='sum').item()
            pixel_count += target.numel()
    finally:
        module.train(was_training)
    if pixel_count == 0:
        raise ValueError('The validation loader must contain mask pixels.')
    return squared_error / pixel_count


def train(opt, args):
    if not hasattr(opt, 'Validation'):
        raise ValueError(
            'Configure Validation.Dataset.type, root and transform_list for '
            'the validation-MSE scheduler. Use deterministic preprocessing '
            'and a validation split separate from the training and test splits.')
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list)

    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    distributed = args.device_num > 1
    validation_loader = None
    if not distributed or dist.get_rank() == 0:
        validation_dataset = eval(opt.Validation.Dataset.type)(
            root=opt.Validation.Dataset.root,
            transform_list=opt.Validation.Dataset.transform_list)
        if len(validation_dataset) == 0:
            raise ValueError('The validation dataset must not be empty.')
        validation_loader = data.DataLoader(
            validation_dataset, batch_size=1, shuffle=False,
            num_workers=opt.Train.Dataloader.num_workers,
            pin_memory=opt.Train.Dataloader.pin_memory)

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.Train.Dataloader.batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=opt.Train.Dataloader.num_workers,
                                    pin_memory=opt.Train.Dataloader.pin_memory,
                                    drop_last=True)

    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 output_stride=opt.Model.output_stride,
                                 pretrained=opt.Model.pretrained)

    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    # Use the same configured learning rate for all trainable parameters.
    optimizer = eval(opt.Train.Optimizer.type)(
        (param for param in model.parameters() if param.requires_grad),
        lr=opt.Train.Optimizer.lr,
        weight_decay=opt.Train.Optimizer.weight_decay)
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    # PyTorch reduces after num_bad_epochs > patience: 19 means 20
    # consecutive epochs without a decrease in validation MSE.
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=19,
        threshold=0.0, threshold_mode='abs')
    model.train()

    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch,
                               position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    sample = to_cuda(sample)
                    out = model(sample)

                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                sample = to_cuda(sample)
                out = model(sample)
                out['loss'].backward()
                optimizer.step()

            if args.local_rank <= 0 and args.verbose is True:
                step_iter.set_postfix({'loss': out['loss'].item()})

        # Validate once per epoch on rank 0, without DDP forward collectives.
        # Broadcast the result so every worker applies the same LR update.
        device = next(model.parameters()).device
        mse = torch.zeros((), dtype=torch.float64, device=device)
        if not distributed or dist.get_rank() == 0:
            mse.fill_(validation_mse(model, validation_loader))
        if distributed:
            dist.broadcast(mse, src=0)
        scheduler.step(mse.item())
        if args.local_rank <= 0 and args.verbose is True:
            epoch_iter.set_postfix({'val_mse': mse.item(),
                                   'lr': optimizer.param_groups[0]['lr']})

        if args.local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(
                ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))

            if args.debug is True:
                debout = debug_tile(out)
                cv2.imwrite(os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(
        ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    train(opt, args)
