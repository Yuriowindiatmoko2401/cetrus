"""
DATSR Streamlit Application
Reference-based Image Super-Resolution with Deformable Attention Transformer
"""

import sys
import os
import streamlit as st
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import components
from core.model_loader import get_model_loader
from core.image_processor import ImageProcessor
from core.inference_engine import InferenceEngine
from core.utils import (
    setup_page_config, add_custom_css, render_system_status,
    create_progress_callback, clear_progress, ErrorHandler,
    save_to_session_state, load_from_session_state
)
from components.uploader import ImageUploader
from components.controls import ControlPanel
from components.viewer import ImageViewer
from config.ui_config import APP_TITLE, APP_DESCRIPTION, DEFAULT_MODEL_TYPE, DEFAULT_SCALE_FACTOR


def main():
    """Main application entry point"""
    # Setup page
    setup_page_config()
    add_custom_css()

    # Render header
    render_header()

    # Initialize components
    initialize_components()

    # Render system status
    render_system_status()

    # Check if models are available
    if not check_models_available():
        st.stop()

    # Main interface
    render_main_interface()


def render_header():
    """Render application header"""
    st.markdown(f"""
    <div class="main-header">
        🖼️ {APP_TITLE}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(APP_DESCRIPTION)


def initialize_components():
    """Initialize application components"""
    if 'components_initialized' not in st.session_state:
        st.session_state.components_initialized = True

        # Initialize core components
        st.session_state.model_loader = get_model_loader()
        st.session_state.image_processor = ImageProcessor(DEFAULT_SCALE_FACTOR)
        st.session_state.inference_engine = InferenceEngine(
            st.session_state.model_loader,
            st.session_state.image_processor
        )

        # Initialize UI components
        st.session_state.uploader = ImageUploader()
        st.session_state.controls = ControlPanel()
        st.session_state.viewer = ImageViewer()


def check_models_available():
    """Check if required model files are available"""
    from core.utils import validate_model_files
    model_status = validate_model_files()

    if not model_status['all_exist']:
        st.error("🚫 预训练模型文件缺失!")
        st.error("请先下载预训练模型文件以使用此应用。")

        with st.expander("📥 下载说明", expanded=True):
            st.markdown("""
            **必需的模型文件:**
            - `feature_extraction.pth`
            - `restoration_mse.pth`
            - `restoration_gan.pth`

            **下载步骤:**
            1. 访问 [DATSR GitHub Releases](https://github.com/caojiezhang/DATSR/releases)
            2. 下载预训练模型文件
            3. 将文件放置在 `DATSR/experiments/pretrained_model/` 目录中

            **目录结构:**
            ```
            DATSR/experiments/pretrained_model/
            ├── feature_extraction.pth
            ├── restoration_mse.pth
            └── restoration_gan.pth
            ```
            """)

        return False

    return True


def render_main_interface():
    """Render the main application interface"""
    # Get uploaded files
    lr_file, ref_file = st.session_state.uploader.get_uploaded_files()

    # Render upload section
    lr_file, ref_file = st.session_state.uploader.render_upload_section()

    # Render controls
    controls = st.session_state.controls.render_sidebar_controls()

    # Render system info
    st.session_state.controls.render_system_info(st.session_state.model_loader)

    # Render tips
    st.session_state.controls.render_processing_tips()

    # Check if ready to process
    ready_to_process = (
        lr_file is not None and
        ref_file is not None and
        controls['process_button']
    )

    if ready_to_process:
        process_images(lr_file, ref_file, controls)
    elif lr_file is not None or ref_file is not None:
        st.info("📤 请上传两张图片以开始处理")

    # Display previous results if available
    display_previous_results()


def process_images(lr_file, ref_file, controls):
    """Process uploaded images with DATSR"""
    try:
        # Show processing started message
        st.info("🚀 开始处理图片...")

        # Create progress callback
        callback, progress_bar, status_text = create_progress_callback()

        # Process images
        results = st.session_state.inference_engine.process_uploaded_images(
            lr_file=lr_file,
            ref_file=ref_file,
            model_type=controls['model_type'],
            scale_factor=controls['scale_factor'],
            progress_callback=callback if controls['advanced_options']['show_progress'] else None
        )

        # Clear progress
        clear_progress(progress_bar, status_text)

        # Save results to session state
        save_to_session_state('latest_results', results)

        # Display results
        if results['success']:
            st.session_state.viewer.render_results_section(results)
        else:
            ErrorHandler.handle_processing_error(results['error'])

    except Exception as e:
        # Clear progress
        if 'progress_bar' in locals():
            clear_progress(progress_bar, status_text)

        ErrorHandler.handle_processing_error(e, show_traceback=True)


def display_previous_results():
    """Display results from previous processing"""
    previous_results = load_from_session_state('latest_results')

    if previous_results and previous_results['success']:
        st.markdown("---")
        st.subheader("📋 上次处理结果")

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🔄 显示上次结果", key="show_previous"):
                save_to_session_state('show_previous_results', True)

        with col2:
            st.info(f"模型: {previous_results['model_type'].upper()} | "
                   f"处理时间: {previous_results['inference_time']:.2f}s | "
                   f"缩放: {previous_results['scale_factor']}x")

        if load_from_session_state('show_previous_results', False):
            st.session_state.viewer.render_results_section(previous_results)


def handle_errors():
    """Global error handler"""
    try:
        main()
    except Exception as e:
        st.error("🚨 应用发生未预期的错误")
        ErrorHandler.handle_processing_error(e, show_traceback=True)

        st.markdown("---")
        st.markdown("### 🔧 故障排除建议")

        suggestions = [
            "刷新页面重试",
            "检查网络连接",
            "确认所有依赖都已正确安装",
            "查看系统状态检查中的错误信息"
        ]

        for suggestion in suggestions:
            st.info(f"• {suggestion}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.info("应用已停止")
    except Exception as e:
        st.error(f"应用启动失败: {str(e)}")
        st.markdown("""
        **可能的解决方案:**
        1. 检查所有依赖是否正确安装
        2. 确认DATSR模块路径正确
        3. 查看终端错误信息获取更多详情

        **安装依赖:**
        ```bash
        pip install -r requirements.txt
        ```
        """)
        # Show full traceback for debugging
        st.markdown("**详细错误信息:**")
        st.code(traceback.format_exc())