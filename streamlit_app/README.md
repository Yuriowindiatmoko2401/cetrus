# DATSR Streamlit Web Interface

A user-friendly web interface for the DATSR (Deformable Attention Transformer for Super-Resolution) project that allows users to upload low-resolution and reference images to generate high-quality super-resolution results.

## Features

- **🖼️ Dual Image Upload**: Upload low-resolution input and reference images
- **🤖 Model Selection**: Choose between MSE (reconstruction-focused) and GAN (perceptual-focused) models
- **⚡ GPU Acceleration**: Automatic GPU detection and fallback to CPU
- **📊 Interactive Results**: Side-by-side comparison, zoom controls, and quality metrics
- **💾 Download Options**: Download individual results or complete ZIP packages
- **🎨 User-Friendly Interface**: Clean, intuitive design with progress indicators

## Installation

### Prerequisites

1. **DATSR Project Setup**: Make sure you have the main DATSR project set up with all dependencies installed as described in the main README.

2. **Download Pretrained Models**: Download the pretrained models from [DATSR GitHub Releases](https://github.com/caojiezhang/DATSR/releases) and place them in:
   ```
   DATSR/experiments/pretrained_model/
   ├── feature_extraction.pth
   ├── restoration_mse.pth
   └── restoration_gan.pth
   ```

### Streamlit App Setup

1. **Install Additional Dependencies**:
   ```bash
   cd /home/yurio/Public/PROJECT/cetrus/streamlit_app
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   # Check that DATSR modules can be imported
   python -c "import sys; sys.path.insert(0, '../DATSR'); from datsr.models import create_model; print('DATSR import successful')"
   ```

## Usage

### Start the Application

```bash
cd /home/yurio/Public/PROJECT/cetrus/streamlit_app
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`.

### How to Use

1. **Upload Images**:
   - **Low-Resolution Input**: The image you want to enhance
   - **Reference Image**: A high-quality image with similar textures/content

2. **Configure Options** (Sidebar):
   - **Model Type**: MSE (better PSNR) or GAN (better visual quality)
   - **Scale Factor**: 2x, 4x, or 8x enlargement
   - **Advanced Options**: GPU/CPU selection, output format, etc.

3. **Process**:
   - Click "🚀 Start Super-Resolution"
   - Monitor progress in real-time
   - View results with interactive comparison tools

4. **Download Results**:
   - Download individual super-resolution result
   - Download bicubic comparison
   - Download complete ZIP package with metadata

### Tips for Best Results

- **Reference Image Selection**: Choose reference images with similar textures and content to your input image
- **Image Quality**: Higher quality reference images generally produce better results
- **Scale Factor**: Higher scale factors (4x, 8x) produce more dramatic enhancement but require more processing time
- **Model Selection**:
  - **MSE Model**: Better for quantitative metrics, more accurate reconstruction
  - **GAN Model**: Better for visual quality, more natural-looking results

## Architecture

### Project Structure

```
streamlit_app/
├── app.py                     # Main Streamlit application
├── config/
│   ├── model_config.py        # Model configuration management
│   └── ui_config.py           # UI settings and constants
├── core/
│   ├── model_loader.py        # Model loading and caching
│   ├── image_processor.py     # Image preprocessing pipeline
│   ├── inference_engine.py    # DATSR inference execution
│   └── utils.py               # Utility functions
├── components/
│   ├── uploader.py            # Image upload interface
│   ├── controls.py            # Processing controls and options
│   └── viewer.py              # Results display and comparison
├── assets/
│   └── example_images/        # Sample images for testing
├── requirements.txt           # Streamlit dependencies
└── README.md                  # This file
```

### Key Components

1. **Model Loader**: Handles loading and caching of DATSR models with device management
2. **Image Processor**: Converts uploaded images to DATSR-compatible tensor format
3. **Inference Engine**: Executes DATSR model inference with progress tracking
4. **UI Components**: Modular Streamlit components for upload, controls, and visualization

## Troubleshooting

### Common Issues

1. **Model Files Missing**:
   - Error: "Missing model files"
   - Solution: Download pretrained models and place in correct directory

2. **CUDA Out of Memory**:
   - Error: "CUDA out of memory"
   - Solution: Use smaller images, switch to CPU mode, or restart the app

3. **Import Errors**:
   - Error: DATSR module import failures
   - Solution: Ensure DATSR dependencies are properly installed and Python path is correct

4. **Image Format Issues**:
   - Error: "Unsupported file type"
   - Solution: Use supported formats (PNG, JPG, JPEG, BMP, TIFF, WEBP)

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.7.1+ (as required by DATSR)
- **GPU**: NVIDIA CUDA-compatible (optional but recommended)
- **RAM**: 8GB+ recommended for large images
- **Storage**: 2GB+ for model files

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is available for 10x+ speedup
- **Image Size**: Larger images require more memory and processing time
- **Model Caching**: Models are cached after first load for faster subsequent processing

## Development

### Running in Development Mode

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Specify port
streamlit run app.py --server.port 8501
```

### Adding New Features

The application is designed with modular components:

1. **New Models**: Update `config/model_config.py` and `core/model_loader.py`
2. **New UI Components**: Add to `components/` directory
3. **New Processing Options**: Extend `core/inference_engine.py`

### Customization

- **UI Styling**: Modify `core/utils.py` add_custom_css() function
- **Default Settings**: Update `config/ui_config.py`
- **Supported Formats**: Modify image validation in components

## License

This Streamlit interface follows the same license as the original DATSR project (CC-BY-NC).

## Acknowledgments

- Original DATSR project: [https://github.com/caojiezhang/DATSR](https://github.com/caojiezhang/DATSR)
- Built with Streamlit: [https://streamlit.io](https://streamlit.io)
- Model architecture based on "Reference-based Image Super-Resolution with Deformable Attention Transformer" (ECCV 2022)