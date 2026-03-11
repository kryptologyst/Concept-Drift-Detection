#!/usr/bin/env python3
"""
Setup script for Concept Drift Detection project.

This script sets up the project environment, installs dependencies,
and runs initial tests to ensure everything is working correctly.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of the command.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("CONCEPT DRIFT DETECTION - PROJECT SETUP")
    print("=" * 60)
    print("⚠️  RESEARCH/EDUCATION ONLY - NOT FOR REGULATED DECISIONS")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "assets",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("❌ Failed to upgrade pip")
        sys.exit(1)
    
    if not run_command("pip install -r requirements.txt", "Installing project dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    print("\nRunning tests...")
    
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Run linting
    print("\nRunning code quality checks...")
    
    if not run_command("python -m black --check src/ tests/ scripts/ demo/", "Checking code formatting"):
        print("⚠️  Code formatting issues found. Run 'black src/ tests/ scripts/ demo/' to fix.")
    
    if not run_command("python -m ruff check src/ tests/ scripts/ demo/", "Running ruff linter"):
        print("⚠️  Linting issues found. Run 'ruff check --fix src/ tests/ scripts/ demo/' to fix.")
    
    # Run the modernized script
    print("\nRunning drift detection demo...")
    
    if not run_command("python 0754_modernized.py", "Running concept drift detection demo"):
        print("❌ Failed to run drift detection demo")
        sys.exit(1)
    
    # Check if assets were created
    if os.path.exists("assets/accuracy_comparison.png"):
        print("✓ Demo completed successfully - assets created")
    else:
        print("⚠️  Demo completed but no assets found")
    
    # Print next steps
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run the Streamlit demo: streamlit run demo/app.py")
    print("2. Run experiments: python scripts/run_experiments.py")
    print("3. Check the DISCLAIMER.md for important limitations")
    print("4. Review the README.md for detailed usage instructions")
    print("\n⚠️  Remember: This is for research/education only!")
    print("=" * 60)


if __name__ == "__main__":
    main()
