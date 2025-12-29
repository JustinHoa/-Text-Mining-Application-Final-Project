# Stop on error
$ErrorActionPreference = "Stop"

$MODEL_BASE = "qwen2.5:3b"
$DECOMPOSER_NAME = "claim-splitter"
$NORMALIZER_NAME = "evidence-normalizer"
$VERIFIER_NAME = "fact-checker"

Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

Write-Host "Pulling base model..."
ollama pull $MODEL_BASE

Write-Host "Creating Decomposer model..."
ollama create $DECOMPOSER_NAME -f ./prompt/decomposer.prompt

Write-Host "Creating Normalizer model..."
ollama create $NORMALIZER_NAME -f ./prompt/normalizer.prompt

Write-Host "Creating Verifier model..."
ollama create $VERIFIER_NAME -f ./prompt/verifier.prompt

Write-Host "Running application..."
python main.py