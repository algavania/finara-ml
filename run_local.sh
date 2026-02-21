#!/bin/bash

# Ensure we're executing in the finara-ml directory, 
# even if the script is called from elsewhere
cd "$(dirname "$0")"

echo "=== Starting Finara ML Locally ==="

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating a new virtual environment (venv)..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip and install dependencies
echo "Installing/verifying dependencies from requirements.txt..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

# 4. Start the server
echo "Starting FastAPI app with uvicorn..."
uvicorn app.main:app --reload
