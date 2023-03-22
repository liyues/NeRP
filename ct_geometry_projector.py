import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch


class Initialization_ConeBeam:
    def __init__(self, image_size, num_proj, start_angle, proj_size, raw_reso=0.7):
        '''
        image_size: [z, x, y], assume x = y for each slice image
        proj_size: [h, w]
        '''
        self.param = {}
        
        self.image_size = image_size
        self.num_proj = num_proj
        self.proj_size = proj_size
        self.raw_reso = raw_reso
        
        self.reso = 512. / image_size[1] * raw_reso

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = image_size[1]
        self.param['ny'] = image_size[2]
        self.param['nz'] = image_size[0]
        self.param['sx'] = self.param['nx']*self.reso
        self.param['sy'] = self.param['ny']*self.reso
        self.param['sz'] = self.param['nz']*self.reso

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle
        self.param['end_angle'] = start_angle + np.pi
        self.param['nProj'] = num_proj

        ## Detector
        self.param['sh'] = self.param['sx']*(1500/1000)
        self.param['sw'] = np.sqrt(self.param['sx']**2+self.param['sy']**2)*(1500/1000)
        self.param['nh'] = proj_size[0] # shape of sinogram is proj_size*proj_size
        self.param['nw'] = proj_size[1]
        self.param['dde'] = 500*self.reso # distance between origin and detector center (assume in x axis)
        self.param['dso'] = 1000*self.reso # distance between origin and source (assume in x axis)

def build_conebeam_gemotry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
                                    max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0, param.param['sz'] / 2.0], 
                                    shape=[param.param['nx'], param.param['ny'], param.param['nz']],
                                    dtype='float32')
    
    angle_partition = odl.uniform_partition(min_pt=param.param['start_angle'], 
                                            max_pt=param.param['end_angle'],
                                            shape=param.param['nProj'])

    detector_partition = odl.uniform_partition(min_pt=[-(param.param['sh'] / 2.0), -(param.param['sw'] / 2.0)], 
                                                 max_pt=[(param.param['sh'] / 2.0), (param.param['sw'] / 2.0)],
                                                 shape=[param.param['nh'], param.param['nw']])

    # Cone-beam geometry for 3D-2D projection
    geometry = odl.tomo.ConeBeamGeometry(apart=angle_partition, # partition of the angle interval
                                          dpart=detector_partition, # partition of the detector parameter interval
                                          src_radius=param.param['dso'], # radius of the source circle
                                          det_radius=param.param['dde'], # radius of the detector circle 
                                          axis=[0, 0, 1]) # rotation axis is z-axis: (0, 0, 1)
    
    ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, # domain of forward projector
                                     geometry=geometry, # geometry of the transform
                                     impl='astra_cuda') # implementation back-end for the transform: ASTRA toolbox, using CUDA, 2D or 3D
    
    FBPOper = odl.tomo.fbp_op(ray_trafo=ray_trafo, 
                             filter_type='Ram-Lak',
                             frequency_scaling=1.0)
    
    # Reconstruction space for imaging object, RayTransform operator, Filtered back-projection operator
    return reco_space, ray_trafo, FBPOper


# Projector
class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        self.reso = param.reso
        
        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        
        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)
        
        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        x = self.trafo(x)
        x = x / self.reso
        return x
    
    def back_projection(self, x):
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_ConeBeam(nn.Module):
    def __init__(self, param):
        super(FBP_ConeBeam, self).__init__()
        self.param = param
        self.reso = param.reso
        
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        
        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class ConeBeam3DProjector():
    def __init__(self, image_size, proj_size, num_proj):

        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle = np.pi / float(self.num_proj * 2.0) * (self.num_proj - 1.0)
        self.raw_reso = 0.7

        # Initialize required parameters for image, view, detector
        geo_param = Initialization_ConeBeam(image_size=self.image_size, 
                                            num_proj=self.num_proj, 
                                            start_angle=self.start_angle,
                                            proj_size=self.proj_size,
                                            raw_reso=self.raw_reso)
        # Forward projection function
        self.forward_projector = Projection_ConeBeam(geo_param)

        # Filtered back-projection
        self.fbp = FBP_ConeBeam(geo_param)

    def forward_project(self, volume):
        '''
        Arguments:
            volume: torch tensor with input size (B, C, img_x, img_y, img_z)
        '''

        proj_data = self.forward_projector(volume)

        return proj_data

    def backward_project(self, projs):
        '''
        Arguments:
            projs: torch tensor with input size (B, num_proj, proj_size_h, proj_size_w)
        '''

        volume = self.fbp(projs)

        return volume














