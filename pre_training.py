import os
import cv2
import time
import yaml
import random
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
from data_util import save_image
from dataset_pretrain import TrainDataset
from model.WAformer import WAformer
from model.generator import G_b, G_d
from loss import L1Loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# set random_seed
setup_seed(2023)


parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', type=str, default='./datasets/ImageNet', help='dataset folder for train (change for different datasets)')
parser.add_argument('--save-epoch-freq', type=int, default=1 , help='how often to save model')
parser.add_argument('--print-freq', type=int, default=20, help='how often to print training information')
parser.add_argument('--train-visual-freq', type=int, default=20, help='how often to visualize training process')
parser.add_argument('--val-visual-freq', type=int, default=30, help='how often to visualize validation process')
parser.add_argument('--resume', type=str, default=None, help='continue training from this checkpoint')
parser.add_argument('--start-epoch', type=int, default=1, help='start epoch')
parser.add_argument('--output-dir', type=str, default='./checkpoints', help='model saved folder')
parser.add_argument('--log-dir', type=str, default='./logs', help='save visual image')
parser.add_argument('--epochs', type=int, default=2, help='total number of epoch')
parser.add_argument('--image-size', type=int, default=128, help='image crop size')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='num of workers per GPU to use')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')


def build_model(opt):

    generator_b = G_b().cuda()
    generator_d = G_d().cuda()

    model = WAformer().cuda()

    optimizer_other = torch.optim.Adam(list(generator_b.parameters())+
                                  list(generator_d.parameters()),
                                  lr=opt.lr,
                                  betas=(0.9, 0.999),eps=1e-8)

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.999), eps=1e-8)

    l1loss = L1Loss().cuda()
    loss = {'l1':l1loss}

    return model, generator_b, generator_d, optimizer_other, optimizer, loss

def get_train_loader(opt):
    train_dataset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)

    return train_dataloader


def load_checkpoint(opt, model, optimizer):
    print(f"=> loading checkpoint '{opt.resume}'")

    checkpoint = torch.load(opt.resume, map_location='cpu')
    opt.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded successfully '{opt.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(epoch, model, optimizer):
    print('==> Saving Epoch: {}'.format(epoch))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(opt.output_dir, 'latest.pth'))



def train(epoch, model, generator_b, generator_d, train_loader, optimizer_other, optimizer, train_loss, opt):

    epoch_start_time = time.time()
    model.train()
    total_epoch_train_loss = []
    for batch_iter, data in enumerate(train_loader):
        image, c_image = data['in_img'].cuda(), data['c_img'].cuda()
        map_b = generator_b(image)
        map_d = generator_d(torch.cat([image,c_image],dim=1))
        input = torch.clip(image * map_b + c_image * map_d, 0, 1)

# =====================WAformer=============================
        optimizer.zero_grad()
        output = model(input.detach())
        losses = train_loss['l1'](output, image.detach())
        losses.backward()
        optimizer.step()

# ==================mu sigma G_b G_d=============================
        optimizer_other.zero_grad()
        output_o = model(input.detach())
        losses_other = -train_loss['l1'](output_o, image.detach())
        losses_other.backward()
        optimizer_other.step()

        total_epoch_train_loss.append(losses.cpu().data)

        if (batch_iter+1) % opt.print_freq == 0:
            print('Epoch: {}, Epoch_iter: {}, Loss: {}'.format(epoch, batch_iter+1, losses))
        if (batch_iter+1) % opt.train_visual_freq == 0:
            print('Saving training image epoch: {}'.format(epoch))
            save_image(epoch, 'train',[image, output, target] , opt)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.epochs, time.time() - epoch_start_time))

    return np.mean(total_epoch_train_loss) / opt.batch_size


def main(opt):

    model, generator_b, generator_d, optimizer_other, optimizer, loss = build_model(opt)
    scheduler.step()

    train_loader = get_trainval_loader(opt)

    total_train_loss = []

    if opt.resume:
        assert os.path.isfile(opt.resume)
        load_checkpoint(opt, model, optimizer_rest)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        epoch_train_loss = train(epoch, model, generator_b, generator_d, train_loader, optimizer_other, optimizer, train_loss, opt)
        total_train_loss.append(epoch_train_loss)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % epoch)
            save_checkpoint(epoch, model, optimizer_rest)

        scheduler.step()

    plt.plot(total_train_loss)
    plt.savefig(opt.log_dir + '/loss.png')

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)

