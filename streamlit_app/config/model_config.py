"""Model configuration for DATSR Streamlit app"""

import os
import sys

# Add parent directories to path to import DATSR modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DATSR'))

def get_model_config(model_type="mse", device="cuda"):
    """Get model configuration for specified model type"""

    base_config = {
        'name': f'test_restoration_{model_type}',
        'suffix': None,
        'scale': 4,
        'model_type': 'RefRestorationModel',
        'set_CUDA_VISIBLE_DEVICES': None,
        'crop_border': None,
        'gpu_ids': [0] if device == 'cuda' else [],

        'network_g': {
            'type': 'SwinUnetv3RestorationNet',
            'ngf': 128,
            'n_blocks': 8,
            'groups': 8,
            'embed_dim': 128,
            'depths': [4, 4],
            'num_heads': [4, 4],
            'window_size': 8,
            'use_checkpoint': True
        },

        'network_map': {
            'type': 'FlowSimCorrespondenceGenerationArch',
            'patch_size': 3,
            'stride': 1,
            'vgg_layer_list': ['relu1_1', 'relu2_1', 'relu3_1'],
            'vgg_type': 'vgg19'
        },

        'network_extractor': {
            'type': 'ContrasExtractorSep'
        },

        'path': {
            'pretrain_model_feature_extractor': 'DATSR/experiments/pretrained_model/feature_extraction.pth',
            'strict_load': True,
            'root': 'experiments/test/'
        }
    }

    # Model-specific paths
    if model_type == "mse":
        base_config['path']['pretrain_model_g'] = 'DATSR/experiments/pretrained_model/restoration_mse.pth'
    elif model_type == "gan":
        base_config['path']['pretrain_model_g'] = 'DATSR/experiments/pretrained_model/restoration_gan.pth'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return base_config

def get_available_models():
    """Get list of available model types"""
    return [
        {"id": "mse", "name": "MSE Model", "description": "Better for reconstruction quality (PSNR-focused)"},
        {"id": "gan", "name": "GAN Model", "description": "Better for perceptual quality (more visually pleasing)"}
    ]

def get_scale_factors():
    """Get available scale factors"""
    return [2, 4, 8]