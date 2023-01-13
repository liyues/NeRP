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

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from ct_geometry_projector import ConeBeam3DProjector
from skimage.measure import compare_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")
parser.add_argument('--iter', type=int, help="load model weights from iter")

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

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

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
model.eval()

# Load pretrain model
model_path = os.path.join(checkpoint_directory, "model_{:06d}.pt".format(opts.iter))
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['net'])
encoder.B = state_dict['enc']
print('Load pretrain model: {}'.format(model_path))

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
    

for it, (grid, image) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()  # [bs, z, x, y, 3], [0, 1]
    image = image.cuda()  # [bs, z, x, y, 1], [0, 1]
    print(grid.shape, image.shape)

    # Data loading
    test_data = (grid, image)  # [bs, z, x, y, 1]

    # Compute testing psnr
    with torch.no_grad():
        test_embedding = encoder.embedding(test_data[0])
        test_output = model(test_embedding)

        test_loss = 0.5 * loss_fn(test_output, test_data[1])
        test_psnr = - 10 * torch.log10(2 * test_loss).item()
        test_loss = test_loss.item()

        test_ssim = compare_ssim(test_output.transpose(1,4).squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)  # [x, y, z] # treat the last dimension of the array as channels

    save_image_3d(test_output, slice_idx, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(opts.iter, test_psnr, test_ssim)))
    print("[Testing Iteration: {}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g}".format(opts.iter, test_loss, test_psnr, test_ssim))
    




