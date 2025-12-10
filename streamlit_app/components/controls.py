"""Control components for DATSR Streamlit app"""

import streamlit as st
from config.model_config import get_available_models, get_scale_factors


class ControlPanel:
    """Handle sidebar controls and advanced options"""

    def render_sidebar_controls(self):
        """Render all sidebar controls"""
        st.sidebar.markdown("## ⚙️ Processing Options")

        # Model selection
        model_type = self._render_model_selection()

        # Scale factor selection
        scale_factor = self._render_scale_selection()

        # Advanced options
        advanced_options = self._render_advanced_options()

        # Processing button
        process_button = self._render_process_button()

        return {
            'model_type': model_type,
            'scale_factor': scale_factor,
            'advanced_options': advanced_options,
            'process_button': process_button
        }

    def _render_model_selection(self):
        """Render model type selection"""
        st.sidebar.markdown("### Model Selection")

        available_models = get_available_models()

        model_options = {model['id']: model['name'] for model in available_models}
        model_descriptions = {model['id']: model['description'] for model in available_models}

        selected_id = st.sidebar.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            key="model_selection"
        )

        # Show model description
        st.sidebar.info(f"ℹ️ {model_descriptions[selected_id]}")

        return selected_id

    def _render_scale_selection(self):
        """Render scale factor selection"""
        st.sidebar.markdown("### Scale Factor")

        scale_factors = get_scale_factors()
        selected_scale = st.sidebar.selectbox(
            "Select Scale Factor:",
            options=scale_factors,
            index=scale_factors.index(4),  # Default to 4x
            key="scale_selection"
        )

        st.sidebar.caption(f"Images will be enlarged {selected_scale}x")

        return selected_scale

    def _render_advanced_options(self):
        """Render advanced options"""
        st.sidebar.markdown("### Advanced Options")

        with st.sidebar.expander("🔧 Advanced Settings"):
            # GPU/CPU selection
            use_gpu = st.checkbox(
                "Use GPU if available",
                value=True,
                key="use_gpu",
                help="Automatically use CUDA if available"
            )

            # Progress display
            show_progress = st.checkbox(
                "Show detailed progress",
                value=True,
                key="show_progress",
                help="Display detailed processing progress"
            )

            # Save intermediate results
            save_intermediate = st.checkbox(
                "Save intermediate results",
                value=False,
                key="save_intermediate",
                help="Save bicubic upscaled version for comparison"
            )

            # Image quality options
            st.markdown("**Output Quality:**")
            output_format = st.selectbox(
                "Output Format:",
                options=["PNG", "JPEG", "WEBP"],
                index=0,
                key="output_format"
            )

            if output_format == "JPEG":
                jpeg_quality = st.slider(
                    "JPEG Quality:",
                    min_value=1,
                    max_value=100,
                    value=95,
                    key="jpeg_quality"
                )
            else:
                jpeg_quality = None

            return {
                'use_gpu': use_gpu,
                'show_progress': show_progress,
                'save_intermediate': save_intermediate,
                'output_format': output_format,
                'jpeg_quality': jpeg_quality
            }

    def _render_process_button(self):
        """Render the main processing button"""
        st.sidebar.markdown("---")

        # Check if files are uploaded
        from components.uploader import ImageUploader
        uploader = ImageUploader()
        lr_file, ref_file = uploader.get_uploaded_files()

        # Enable/disable button based on file availability
        button_disabled = lr_file is None or ref_file is None

        if button_disabled:
            st.sidebar.warning("⚠️ Please upload both images first")
            process_button = st.sidebar.button(
                "🚀 Start Super-Resolution",
                disabled=True,
                key="process_button"
            )
        else:
            st.sidebar.success("✅ Ready to process!")
            process_button = st.sidebar.button(
                "🚀 Start Super-Resolution",
                type="primary",
                key="process_button"
            )

        return process_button

    def render_system_info(self, model_loader):
        """Render system information panel"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 System Information")

        # Get device info
        device_info = model_loader.get_device_info()

        st.sidebar.metric("Device", device_info['current_device'].upper())

        if device_info['cuda_available']:
            if device_info['cuda_device_count'] > 0:
                st.sidebar.metric("GPU", device_info['cuda_device_name'])
                if device_info['cuda_memory_allocated'] > 0:
                    memory_mb = device_info['cuda_memory_allocated'] / (1024**2)
                    st.sidebar.metric("GPU Memory Used", f"{memory_mb:.1f} MB")
        else:
            st.sidebar.info("CUDA not available, using CPU")

        # Model status
        st.sidebar.markdown("### 🤖 Model Status")
        model_info = model_loader.get_model_info("mse")  # Check MSE model as example

        if model_info['status'] == 'loaded':
            st.sidebar.success("✅ Model Loaded")
            if model_info['parameters'] > 0:
                params_m = model_info['parameters'] / 1e6
                st.sidebar.caption(f"Parameters: {params_m:.1f}M")
        else:
            st.sidebar.info("Model not loaded (will load on demand)")

    def render_processing_tips(self):
        """Render helpful tips for users"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Tips")

        tips = [
            "For best results, use a reference image with similar content and textures",
            "Higher scale factors require more processing time and memory",
            "GPU processing is significantly faster than CPU",
            "Results may vary depending on image quality and compatibility"
        ]

        for tip in tips:
            st.sidebar.info(f"• {tip}")

        # Help section
        with st.sidebar.expander("❓ Need Help?"):
            st.markdown("""
            **How to use:**
            1. Upload a low-resolution image
            2. Upload a high-quality reference image
            3. Select model and scale factor
            4. Click "Start Super-Resolution"

            **Best Practices:**
            - Use reference images with similar textures
            - Try both MSE and GAN models for different results
            - MSE models are better for quantitative metrics
            - GAN models often produce more visually pleasing results
            """)