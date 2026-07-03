#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}▸ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

echo "============================================"
echo "  Ops Agent — Setup"
echo "============================================"

step "Checking Python 3.10+..."
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install Python 3.10+ first."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found Python $PY_VER"

step "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    fail "Ollama not found. Install from https://ollama.ai first."
fi
echo "  Ollama is installed"

if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    warn "Ollama doesn't seem to be running. Starting it..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Ollama 2>/dev/null || warn "Could not auto-start Ollama. Please start it manually."
        sleep 3
    else
        warn "Please start Ollama manually (run 'ollama serve' in another terminal)"
        read -p "  Press Enter when Ollama is running..."
    fi
fi

step "Pulling LLM models (this may take a few minutes)..."

pull_if_missing() {
    local model=$1
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo "  ✓ $model already available"
    else
        echo "  Pulling $model..."
        ollama pull "$model"
    fi
}

pull_if_missing "gemma3:4b"
pull_if_missing "all-minilm"

echo ""
read -p "  Pull gemma3:1b for baseline comparison? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pull_if_missing "gemma3:1b"
fi

read -p "  Pull qwen2.5:7b for better accuracy? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pull_if_missing "qwen2.5:7b"
fi

step "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists"
fi
source venv/bin/activate
echo "  Activated venv"

step "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "  Installed $(pip list --format=columns 2>/dev/null | wc -l | tr -d ' ') packages"

step "Checking .env configuration..."
if [ ! -f ".env" ]; then
    if [ -f "app/.env" ]; then
        cp app/.env .env
        warn "Copied app/.env to project root"
    else
        fail ".env file not found. Create one from the README."
    fi
fi
echo "  .env exists"
echo "  LLM_MODEL=$(grep '^LLM_MODEL=' .env | cut -d= -f2)"

step "Ensuring data directories exist..."
mkdir -p data/eval_results data/runbooks
echo "  data/ directories ready"

step "Starting server for smoke test..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 &>/dev/null &
SERVER_PID=$!
sleep 3

if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
    echo "  Server is healthy"
else
    kill $SERVER_PID 2>/dev/null || true
    fail "Server failed to start. Check the logs."
fi

step "Ingesting runbooks for RAG..."
INGEST=$(curl -s -X POST http://127.0.0.1:8000/runbooks/ingest)
echo "  $INGEST"

step "Running test question: 'Why is service-x slow?'"
RESULT=$(curl -s -X POST http://127.0.0.1:8000/run \
    -H "Content-Type: application/json" \
    -d '{"question": "Why is service-x slow?"}')

TOOLS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tools_used',[]))" 2>/dev/null || echo "[]")
echo "  Tools used: $TOOLS"

ANSWER_LEN=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('final_answer','')))" 2>/dev/null || echo "0")
echo "  Answer length: $ANSWER_LEN chars"

kill $SERVER_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Start the server:"
echo "     source venv/bin/activate"
echo "     uvicorn app.main:app --reload"
echo ""
echo "  2. Run the model comparison eval:"
echo "     cd app/eval"
echo "     python run_eval.py --compare gemma3:1b gemma3:4b"
echo ""
echo "  3. View results:"
echo "     python run_eval.py --report"
echo ""
