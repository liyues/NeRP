import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX

import numpy as np
from tqdm import tqdm

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from ct_geometry_projector import ConeBeam3DProjector
from skimage.measure import compare_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
if opts.pretrain: 
    output_subfolder = config['data'] + '_pretrain'
else:
    output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
    .format(config['img_size'], config['num_proj'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
if config['model'] == 'SIREN':
    model = SIREN(config['net'])
elif config['model'] == 'FFN':
    model = FFN(config['net'])
else:
    raise NotImplementedError
model.cuda()
model.train()

# Load pretrain model
if opts.pretrain:
    model_path = config['pretrain_model_path'].format(config['img_size'], \
                    config['model'], config['net']['network_input_size'], config['net']['network_width'], \
                    config['net']['network_depth'], config['encoder']['scale'], config['encoder']['embedding_size'])
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['net'])
    encoder.B = state_dict['enc']
    print('Load pretrain model: {}'.format(model_path))

# Setup optimizer
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
else:
    NotImplementedError

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])

config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))
if config['num_proj'] > config['display_image_num']:
    proj_idx = list(range(0, config['num_proj'], int(config['num_proj']/config['display_image_num'])))
else:
    proj_idx = list(range(0, config['num_proj']))

ct_projector = ConeBeam3DProjector(config['img_size'], config['proj_size'], config['num_proj'])

for it, (grid, image) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()  # [bs, z, x, y, 3], [0, 1]
    image = image.cuda()  # [bs, z, x, y, 1], [0, 1]
    print(grid.shape, image.shape)

    projs = ct_projector.forward_project(image.transpose(1, 4).squeeze(1))  # [bs, x, y, z] -> [bs, n, h, w]
    print(projs.shape)

    # FBP recon
    fbp_recon = ct_projector.backward_project(projs)  # [bs, n, h, w] -> [bs, x, y, z]
    print(fbp_recon.shape)


    # Data loading
    test_data = (grid, image)  # [bs, z, x, y, 1]
    train_data = (grid, projs)  # [bs, n, h, w]

    save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
    save_image_3d(train_data[1].transpose(2, 3).unsqueeze(-1), proj_idx, os.path.join(image_directory, "train.png"))
    
    fbp_recon_ssim = compare_ssim(fbp_recon.squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)  # [x, y, z] # treat the last dimension of the array as channels
    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)  # [bs, z, x, y, 1]
    fbp_recon_psnr = - 10 * torch.log10(loss_fn(fbp_recon, test_data[1]))
    save_image_3d(fbp_recon, slice_idx, os.path.join(image_directory, "fbp_recon_{:.4g}dB_ssim{:.4g}.png".format(fbp_recon_psnr, fbp_recon_ssim)))

    # Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()

        train_embedding = encoder.embedding(train_data[0])  # [bs, z, x, y, embedding*2]
        train_output = model(train_embedding)

        train_projs = ct_projector.forward_project(train_output.transpose(1, 4).squeeze(1))
        train_loss = 0.5 * loss_fn(train_projs, train_data[1])

        train_loss.backward()
        optim.step()

        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()

            train_writer.add_scalar('train_loss', train_loss, iterations + 1)
            train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))

        # Compute testing psnr
        if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)

                test_loss = 0.5 * loss_fn(test_output, test_data[1])
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()

                test_ssim = compare_ssim(test_output.transpose(1,4).squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)

            train_writer.add_scalar('test_loss', test_loss, iterations + 1)
            train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
            save_image_3d(test_output, slice_idx, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iterations + 1, test_psnr, test_ssim)))
            print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr, test_ssim))
        
        # Save final model
        if (iterations + 1) % config['image_save_iter'] == 0:
            model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
            torch.save({'net': model.state_dict(), \
                        'enc': encoder.B, \
                        'opt': optim.state_dict(), \
                        }, model_name)




