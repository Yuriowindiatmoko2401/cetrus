#!/usr/bin/env python3
"""
Test script to validate Streamlit app structure and basic functionality
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_structure():
    """Check if all required directories and files exist"""
    print("🔍 Checking directory structure...")

    required_dirs = [
        "config",
        "core",
        "components",
        "assets",
        "assets/example_images"
    ]

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "config/__init__.py",
        "config/model_config.py",
        "config/ui_config.py",
        "core/__init__.py",
        "core/model_loader.py",
        "core/image_processor.py",
        "core/inference_engine.py",
        "core/utils.py",
        "components/__init__.py",
        "components/uploader.py",
        "components/controls.py",
        "components/viewer.py"
    ]

    # Check directories
    for dirname in required_dirs:
        if os.path.exists(dirname):
            print(f"✅ Directory: {dirname}")
        else:
            print(f"❌ Directory: {dirname} (MISSING)")

    # Check files
    for filename in required_files:
        check_file_exists(filename, f"File")

    print()

def check_python_syntax():
    """Check Python syntax for all Python files"""
    print("🐍 Checking Python syntax...")

    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                python_files.append(os.path.join(root, file))

    errors = []
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✅ Syntax OK: {filepath}")
        except SyntaxError as e:
            print(f"❌ Syntax Error in {filepath}: {e}")
            errors.append((filepath, str(e)))
        except Exception as e:
            print(f"⚠️  Could not check {filepath}: {e}")

    if errors:
        print(f"\n❌ Found {len(errors)} syntax errors:")
        for filepath, error in errors:
            print(f"  • {filepath}: {error}")
    else:
        print("\n✅ All Python files have valid syntax!")

    return len(errors) == 0

def check_imports():
    """Check if basic imports work (without DATSR dependencies)"""
    print("📦 Checking basic imports...")

    try:
        # Check if we can import config modules
        sys.path.insert(0, '.')
        from config.model_config import get_model_config, get_available_models
        print("✅ Config modules import successfully")

        # Test basic functionality
        models = get_available_models()
        print(f"✅ Available models: {[model['id'] for model in models]}")

        config = get_model_config('mse', 'cpu')
        print("✅ Model configuration generation works")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_datrs_integration():
    """Check if DATSR path and structure looks correct"""
    print("🔗 Checking DATSR integration...")

    datsr_path = "../DATSR"
    if not os.path.exists(datsr_path):
        print(f"❌ DATSR directory not found: {datsr_path}")
        return False

    print(f"✅ DATSR directory found: {datsr_path}")

    # Check key DATSR directories
    key_dirs = ["datsr", "experiments", "options"]
    for dirname in key_dirs:
        dir_path = os.path.join(datsr_path, dirname)
        if os.path.exists(dir_path):
            print(f"✅ DATSR {dirname}/ directory exists")
        else:
            print(f"❌ DATSR {dirname}/ directory missing")

    # Check for model directories
    model_dir = os.path.join(datsr_path, "experiments", "pretrained_model")
    if os.path.exists(model_dir):
        print(f"✅ Model directory exists: {model_dir}")

        # List existing model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if model_files:
            print(f"✅ Found model files: {model_files}")
        else:
            print("⚠️  No model files found (need to download pretrained models)")
    else:
        print(f"⚠️  Model directory not found (will be created): {model_dir}")

    return True

def main():
    """Run all tests"""
    print("🚀 DATSR Streamlit App Structure Test")
    print("=" * 50)

    all_good = True

    # Check directory structure
    check_directory_structure()

    # Check Python syntax
    syntax_ok = check_python_syntax()
    all_good = all_good and syntax_ok

    # Check basic imports
    imports_ok = check_imports()
    all_good = all_good and imports_ok

    # Check DATSR integration
    datrs_ok = check_datrs_integration()
    all_good = all_good and datrs_ok

    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All tests passed! The Streamlit app structure is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download pretrained models to DATSR/experiments/pretrained_model/")
        print("3. Run the app: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")

    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)