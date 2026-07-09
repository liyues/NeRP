# Changes For Running

## utils.py
```python

# Line 13
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream) # change to safe load

# Line 30
def get_data_loader(data, img_path, img_dim, img_slice,
                    train, batch_size, 
                    num_workers=0, # Number of workers 0 for windows only
                    return_data_idx=False):

# Line 93
def ct_parallel_project_2d(img, theta):
	bs, h, w, c = img.size()

	# (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
	y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5,
							torch.arange(w, dtype=torch.float32) / w - 0.5])
    
	x = x.to(theta.device) # set device to gpu
	y = y.to(theta.device) # set device to gpu
    
	# Rotation transform matrix: simulate parallel projection rays
	x_rot = x * torch.cos(theta) - y * torch.sin(theta)
	y_rot = x * torch.sin(theta) + y * torch.cos(theta)

	# Reverse back to index [0, w]
	x_rot = (x_rot + 0.5) * w
	y_rot = (y_rot + 0.5) * h

	# Resample (x, y) index of the pixel on the projection ray-theta
	sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
	img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

	# Compute integral projections along rays
	proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]

	return proj

```

## Config Section 3D
```python

# cr_recon_3d.yaml
# network_input_size: 256 
# network_depth: 6
# network_width: 128
# embedding_size: 128 

# image_regression_3d.yaml
# network_input_size: 256     
# network_output_size: 1
# network_depth: 6            
# network_width: 128
# embedding_size: 128

```

## train_ct_recon_3d.py
```python

# Line 122
fbp_recon_ssim = compare_ssim(
        fbp_recon.squeeze().cpu().numpy(), 
        test_data[1].transpose(1,4).squeeze().cpu().numpy(), 
        data_range=1.0, 
        channel_axis=-1
    )

# Line 166
test_ssim = compare_ssim(
                    test_output.transpose(1,4).squeeze().cpu().numpy(), 
                    test_data[1].transpose(1,4).squeeze().cpu().numpy(), 
                    data_range=1.0, 
                    channel_axis=-1
                )

```

## \NeRP\venv\lib\site-packages\odl\contrib\torch\operator.py
```python

# Change all AVOID_UNNECESSARY_COPY to False

```