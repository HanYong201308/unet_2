import os
import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset

from modules.loss import CEDiceLoss, BCEDiceLoss
from modules.datasets import Segmentation_dataset
from modules.transforms import original_transform, teacher_transform
from models.unet import UNet
from models.fcn import fcn8s
from models.segnet import segnet
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /unet


dataroot = os.path.join(os.path.dirname(base_path), "datasets")
if not os.path.exists(dataroot):
    os.mkdir(dataroot)

datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=True, transform=original_transform, target_transform=teacher_transform)

train_loader = torch.utils.data.DataLoader(datasets, batch_size=4, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_fcn8 = fcn8s().to(device)
criterion_fcn8 = BCEDiceLoss().to(device)



optimizer_fcn8 = optim.Adam(model_fcn8.parameters(), lr=0.001)

# colabは相対パスがいいみたい
# logdir = "logs"
# logdir_path = os.path.join(base_path, logdir)
logdir_path = "./logs"
if not os.path.isdir(logdir_path):
    os.mkdir(logdir_path)
dt = datetime.datetime.now()
model_id = len(glob.glob(os.path.join(logdir_path, "{}{}{}*".format(dt.year, dt.month, dt.day))))
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer = SummaryWriter(log_dir=log_path)

epochs = 2


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model_fcn8.train()

        optimizer_fcn8.zero_grad()

        adjust_learning_rate(optimizer_fcn8, epoch)

        # data, target = data.cuda(), target.cuda()

        output_fcn8 = model_fcn8(data)
        loss_fcn8 = criterion_fcn8(output_fcn8, target)
        loss_fcn8.backward()
        optimizer_fcn8.step()

        writer.add_scalar("train_loss for fcn8", loss_fcn8.item(), (len(train_loader)*(epoch-1)+batch_idx)) # 675*e+i

        if batch_idx % 20 == 0:
            #validation
            # model.eval()
            # with torch.no_grad():
            #     val_losses = 0.0
            #     for idx, (data, target) in enumerate(val_loader):
            #         data, target = data.cuda(), target.cuda()
            #         embedded = model(data)
            #         val_loss = criterion(embedded, target)
            #         val_losses += val_loss
            # mean_val_loss = val_losses/len(val_loader)
            # writer.add_scalar("loss/val_loss", loss.item(), (len(train_loader)*(epoch-1)+batch_idx))
            # print('Train Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain_loss: {:>2.4f}\tval_loss: {:>2.4f}'.format(
            #     epoch,
            #     batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            #     loss.item(), val_loss))
            print('Train Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain_loss for '
                  'fcn8: {:>2.4f}'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss_fcn8.item()))


def save(epoch):
    checkpoint_path = os.path.join(base_path, "checkpoints")
    save_file = "checkpoint.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model_fcn8.state_dict(), save_path)


if __name__ == "__main__":
    model_load = False
    if model_load == True:
        start_epoch = 52
        epoch_range = range(start_epoch, epochs+1)
    else:
        epoch_range = range(1, epochs+1)

    for epoch in epoch_range:
        train(epoch)
        save(epoch)
