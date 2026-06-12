#!/usr/bin/env bash
# init.sh — GraphRAG Code Review Bot environment setup
# Run this before starting the server for the first time, or after pulling changes.

set -e

# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${BLUE}→${NC} $1"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   GraphRAG Code Review Bot — Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Check Python version ────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3 || command -v python || fail "Python not found in PATH")
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    fail "Python 3.9+ required, found $PY_VERSION"
fi
ok "Python $PY_VERSION"

# ── 2. Load .env if present ────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    ok "Loaded .env"
else
    warn ".env not found — copy .env-example and fill in values"
fi

# ── 3. Check GROQ_API_KEY ─────────────────────────────────────────────────────
info "Checking GROQ_API_KEY..."
if [ -z "${GROQ_API_KEY:-}" ]; then
    fail "GROQ_API_KEY is not set. Add it to .env (get a free key at https://console.groq.com)"
fi
ok "GROQ_API_KEY is set (${#GROQ_API_KEY} chars)"

# ── 4. Check Neo4j connectivity ───────────────────────────────────────────────
NEO4J_URL="${APP_NEO4J_URL:-bolt://localhost:7687}"
NEO4J_USER="${APP_NEO4J_USER:-neo4j}"
NEO4J_PASS="${APP_NEO4J_PASSWORD:-}"
info "Checking Neo4j at $NEO4J_URL..."
if [ -z "$NEO4J_PASS" ]; then
    warn "APP_NEO4J_PASSWORD is not set. Neo4j check skipped."
else
    $PYTHON -c "
from neo4j import GraphDatabase
try:
    d = GraphDatabase.driver('$NEO4J_URL', auth=('$NEO4J_USER', '$NEO4J_PASS'))
    d.verify_connectivity()
    d.close()
    print('ok')
except Exception as e:
    print(f'error: {e}')
    exit(1)
" && ok "Neo4j connection verified" || fail "Cannot connect to Neo4j at $NEO4J_URL. Is Neo4j running?"
fi

# ── 5. Install Python dependencies ────────────────────────────────────────────
info "Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt --quiet && ok "Python dependencies installed"

# ── 6. Create data directory ──────────────────────────────────────────────────
info "Ensuring data/ directory exists..."
mkdir -p data/projects
ok "data/projects/ ready"

# ── 7. Initialize SQLite database ────────────────────────────────────────────
info "Initializing SQLite database..."
$PYTHON -c "
import asyncio
import sys
sys.path.insert(0, '.')
from api.database import init_db
asyncio.run(init_db())
print('ok')
" && ok "SQLite database initialized"

# ── 8. Check frontend dependencies ───────────────────────────────────────────
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    if [ ! -d "frontend/node_modules" ]; then
        info "Installing frontend dependencies..."
        (cd frontend && npm install --silent) && ok "Frontend dependencies installed"
    else
        ok "Frontend node_modules present"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  Setup complete!${NC}"
echo ""
echo "  Start backend:   uvicorn api.main:app --reload --port 8000"
echo "  Start frontend:  cd frontend && npm run dev"
echo "  Run tests:       pytest tests/ -v --tb=short"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
