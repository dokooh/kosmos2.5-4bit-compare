#!/bin/bash
# Setup script for Kosmos 2.5 quantization testing

echo "Setting up Kosmos 2.5 quantization testing environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To use the script:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the script: python kosmos2_5_quantization_test.py <input_image> <output_file>"
echo ""
echo "Example: python kosmos2_5_quantization_test.py test_image.jpg results.json"