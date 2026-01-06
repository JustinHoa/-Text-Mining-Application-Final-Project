# Stop on error
$ErrorActionPreference = "Stop"

Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

Write-Host "Running seeding..."
python data/seeding.py

# Write-Host "Running embedding..."
# python data/embedding.py

Write-Host "Running retrieving for testing..."
python data/retrieving.py

Write-Host "Setup completed."