#!/usr/bin/env python3
"""
Simple validation script to check the project structure and code quality.
This can run without installing all dependencies.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ Missing {description}: {filepath}")
        return False

def check_python_syntax(filepath):
    """Check Python syntax without importing modules."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"✓ Syntax OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax Error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 Validating CNN Binary Classifier Project")
    print("=" * 50)
    
    # Check essential files
    essential_files = [
        ("requirements.txt", "Dependencies file"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("README.md", "Documentation"),
        ("app.py", "Streamlit application"),
        ("train.py", "Training script"),
        ("src/model.py", "CNN model definition"),
        ("src/__init__.py", "Source package init"),
        ("create_demo_model.py", "Demo model creator")
    ]
    
    print("\n📁 Checking file structure:")
    all_files_exist = True
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    # Check Python syntax
    python_files = [
        "app.py",
        "train.py", 
        "src/model.py",
        "src/__init__.py",
        "create_demo_model.py"
    ]
    
    print("\n🐍 Checking Python syntax:")
    all_syntax_ok = True
    for filepath in python_files:
        if os.path.exists(filepath):
            if not check_python_syntax(filepath):
                all_syntax_ok = False
    
    # Check directory structure
    print("\n📂 Checking directory structure:")
    required_dirs = ["src", "models", "data"]
    all_dirs_exist = True
    for dirname in required_dirs:
        if os.path.exists(dirname):
            print(f"✓ Directory exists: {dirname}/")
        else:
            print(f"✗ Missing directory: {dirname}/")
            all_dirs_exist = False
    
    # Check configuration files
    print("\n⚙️ Checking configuration files:")
    
    # Check requirements.txt content
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
            required_packages = ["torch", "streamlit", "pillow", "numpy"]
            for package in required_packages:
                if package in requirements:
                    print(f"✓ Required package found: {package}")
                else:
                    print(f"✗ Missing required package: {package}")
    
    # Check Dockerfile content
    if os.path.exists("Dockerfile"):
        with open("Dockerfile", 'r') as f:
            dockerfile = f.read()
            if "FROM python:" in dockerfile:
                print("✓ Dockerfile has Python base image")
            if "EXPOSE 8501" in dockerfile:
                print("✓ Dockerfile exposes Streamlit port")
            if "streamlit run app.py" in dockerfile:
                print("✓ Dockerfile runs Streamlit app")
    
    # Summary
    print("\n" + "=" * 50)
    if all_files_exist and all_syntax_ok and all_dirs_exist:
        print("🎉 PROJECT VALIDATION PASSED!")
        print("The CNN binary classifier project is properly structured.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create a demo model: python create_demo_model.py")
        print("3. Run the app: streamlit run app.py")
        print("4. Or use Docker: docker-compose up --build")
        return True
    else:
        print("❌ PROJECT VALIDATION FAILED!")
        print("Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)