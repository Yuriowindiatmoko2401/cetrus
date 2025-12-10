"""Model loading and caching for DATSR Streamlit app"""

import os
import sys
import torch
from typing import Dict, Any, Optional

# Add DATSR path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DATSR'))

try:
    from datsr.models import create_model
    from datsr.utils.options import dict_to_nonedict
    from config.model_config import get_model_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure DATSR modules are available in the Python path")
    create_model = None
    dict_to_nonedict = None


class ModelLoader:
    """Handle DATSR model loading and caching"""

    def __init__(self):
        self.loaded_models = {}
        self.device = self._get_device()

    def _get_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def load_model(self, model_type: str = "mse", force_reload: bool = False) -> Any:
        """
        Load and cache DATSR model

        Args:
            model_type: Type of model to load ("mse" or "gan")
            force_reload: Whether to force reload even if cached

        Returns:
            Loaded DATSR model
        """
        if create_model is None:
            raise ImportError("DATSR model creation function not available")

        # Check cache
        if not force_reload and model_type in self.loaded_models:
            return self.loaded_models[model_type]

        try:
            # Get model configuration
            opt = get_model_config(model_type, self.device)
            opt = dict_to_nonedict(opt)

            # Set GPU IDs based on device
            if self.device == 'cuda':
                opt['gpu_ids'] = [0]
            else:
                opt['gpu_ids'] = []

            print(f"Loading {model_type} model on {self.device}...")

            # Check if model files exist
            self._check_model_files(opt)

            # Create model
            model = create_model(opt)

            # Set to evaluation mode
            if hasattr(model, 'eval'):
                model.eval()
            elif hasattr(model, 'net_g'):
                model.net_g.eval()

            # Cache the model
            self.loaded_models[model_type] = model

            print(f"Successfully loaded {model_type} model")
            return model

        except Exception as e:
            error_msg = f"Failed to load {model_type} model: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)

    def _check_model_files(self, opt: Dict[str, Any]):
        """Check if required model files exist"""
        missing_files = []

        # Check main model
        main_model_path = opt['path']['pretrain_model_g']
        if not os.path.exists(main_model_path):
            missing_files.append(main_model_path)

        # Check feature extractor
        feature_extractor_path = opt['path']['pretrain_model_feature_extractor']
        if not os.path.exists(feature_extractor_path):
            missing_files.append(feature_extractor_path)

        if missing_files:
            error_msg = f"Missing model files: {missing_files}\n"
            error_msg += "Please download pretrained models and place them in the correct directory.\n"
            error_msg += "See DATSR/README.md for download instructions."
            raise FileNotFoundError(error_msg)

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_type not in self.loaded_models:
            return {"status": "not_loaded", "device": None}

        model = self.loaded_models[model_type]
        return {
            "status": "loaded",
            "device": self.device,
            "model_type": model_type,
            "parameters": self._count_parameters(model)
        }

    def _count_parameters(self, model) -> int:
        """Count total parameters in model"""
        try:
            if hasattr(model, 'net_g'):
                return sum(p.numel() for p in model.net_g.parameters())
            else:
                return sum(p.numel() for p in model.parameters())
        except:
            return 0

    def clear_cache(self):
        """Clear model cache"""
        self.loaded_models.clear()

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices"""
        device_info = {
            "current_device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count()
        }

        if torch.cuda.is_available():
            device_info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                "cuda_memory_allocated": torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0,
                "cuda_memory_reserved": torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else 0
            })

        return device_info

    def unload_model(self, model_type: str):
        """Unload a specific model from cache"""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]

            # Clear CUDA cache if needed
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def preload_models(self):
        """Preload all available models"""
        model_types = ["mse", "gan"]
        for model_type in model_types:
            try:
                self.load_model(model_type)
            except Exception as e:
                print(f"Failed to preload {model_type} model: {e}")


# Global model loader instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def load_datsr_model(model_type: str = "mse", device: Optional[str] = None) -> Any:
    """Convenience function to load a DATSR model"""
    loader = get_model_loader()

    if device:
        loader.device = device

    return loader.load_model(model_type)