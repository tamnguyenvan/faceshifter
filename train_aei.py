"""
"""
import os
import time
import argparse

import cv2
import visdom
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader 
# from apex import amp

from models.aei_net import AEINet
from models.multiscale_discriminator import MultiscaleDiscriminator
from datasets.base import FaceEmbeddingDataset
from external.arcface_torch.backbones import get_model


def hinge_loss(X, positive=True):
    """
    """
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


def get_grid_image(X):
    """
    """
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    """
    """
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()


def train():
    # Settings
    # vis = visdom.Visdom(server='127.0.0.1', env='faceshifter', port=8099)
    batch_size = args.batch_size
    lr_generator = args.g_lr
    lr_discriminator = args.d_lr
    max_epoch = args.epochs
    arcface_backbone = args.arcface_backbone
    gpus = args.gpus.split(',')
    n_gpus = len(gpus)
    show_step = 10
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Build ArcFace model
    print('Creating ArcFace model...')
    arcface = get_model(arcface_backbone, fp16=False)
    arcface.eval()
    arcface.load_state_dict(torch.load(args.arcface_weights, map_location=device), strict=False)

    # Build AEI model
    print('Creating AEI model...')
    generator = AEINet(num_features=arcface.num_features).to(device)
    discriminator = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
    generator.train()
    discriminator.train()

    gen_optimizer = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0, 0.999))

    if n_gpus > 1:
        generator, gen_optimizer = amp.initialize(generator, gen_optimizer)
        discriminator, disc_optimizer = amp.initialize(discriminator, disc_optimizer)

    gen_ckpt_path = os.path.join(args.ckpt_dir, 'generator_latest.pth')
    disc_ckpt_path = os.path.join(args.ckpt_dir, 'discriminator_latest.pth')
    if os.path.isfile(gen_ckpt_path):
        generator.load_state_dict(torch.load(gen_ckpt_path, map_location=torch.device('cpu')), strict=False)
    if os.path.isfile(disc_ckpt_path):
        discriminator.load_state_dict(torch.load(disc_ckpt_path, map_location=torch.device('cpu')), strict=False)

    # Face embedding model
    dataset = FaceEmbeddingDataset(['data/ffhq'], same_prob=0.8)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    for epoch in range(0, max_epoch):
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            Xs, Xt, same_person = data
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            same_person = same_person.to(device)

            # Identity encoder
            with torch.no_grad():
                embed = arcface(F.interpolate(Xs, [112, 112], mode='bilinear', align_corners=True))

            # Train generator
            gen_optimizer.zero_grad()
            Y, Xt_attr = generator(Xt, embed)

            disc_outputs = discriminator(Y)
            L_adv = 0
            for di in disc_outputs:
                L_adv += hinge_loss(di[0], True)
            
            Y_aligned = Y
            ZY = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
            L_id = (1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

            Y_attr = generator.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0
            L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

            lossgenerator = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
            if n_gpus > 1:
                with amp.scale_loss(lossgenerator, gen_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossgenerator.backward()

            gen_optimizer.step()

            # train discriminator
            disc_optimizer.zero_grad()
            fake_discriminator = discriminator(Y.detach())
            loss_fake = 0
            for di in fake_discriminator:
                loss_fake += hinge_loss(di[0], False)

            true_discriminator = discriminator(Xs)
            loss_true = 0
            for di in true_discriminator:
                loss_true += hinge_loss(di[0], True)

            lossdiscriminator = 0.5*(loss_true.mean() + loss_fake.mean())

            if n_gpus > 1:
                with amp.scale_loss(lossdiscriminator, disc_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossdiscriminator.backward()

            disc_optimizer.step()
            batch_time = time.time() - start_time
            if (iteration + 1) % show_step == 0:
                image = make_image(Xs, Xt, Y)
                vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
                cv2.imwrite('./gen_images/latest.jpg', image.transpose([1,2,0]))

            print(f'[Epoch {epoch}/{max_epoch}, iter {iteration}/{len(dataloader)}] - '
                  f'discriminator_loss: {lossdiscriminator.item()} - generator_loss: {lossgenerator.item()} - Time: {batch_time}s - '
                  f'L_adv: {L_adv.item()} -  L_id: {L_id.item()} - L_attr: {L_attr.item()} - L_rec: {L_rec.item()}')
            if (iteration + 1) % 1000 == 0:
                torch.save(generator.state_dict(), gen_ckpt_path)
                torch.save(discriminator.state_dict(), disc_ckpt_path)
                print(f'Saved generator as {gen_ckpt_path}')
                print(f'Saved discriminator as {disc_ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--g-lr', type=float, default=4e4,
                        help='generatorenerator learning rate')
    parser.add_argument('--d-lr', type=float, default=4e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Number of gpus will be used. e.g --gpus 0,1,2')
    parser.add_argument('--ckpt-dir', type=str, default='weights',
                        help='Checkpoint directory')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Logs directory')
    parser.add_argument('--arcface-backbone', type=str, default='r50',
                        help='ArcFace backbone')
    parser.add_argument('--arcface-weights', type=str, default='external/weights/arcface_r50_backbone.pth',
                        help='Path to ArcFace backbone weights file')

    args = parser.parse_args()
    print(args)
    train()