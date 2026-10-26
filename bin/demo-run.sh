#!/usr/bin/env bash
# ==============================================================================
# Pinecone Vector Forge - 60-Second Full Tour Runner
# ==============================================================================
set -e

GREEN="\033[0;32m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}   🚀 Pinecone Vector Forge — Automated Showcase Tour            ${NC}"
echo -e "${CYAN}=================================================================${NC}"

# Detect architecture
ARCH_CMD="arch -arm64"
if [ "$(uname -m)" != "arm64" ]; then
    ARCH_CMD=""
fi

echo -e "\n${GREEN}[1/4] Running Comprehensive Verification Test Suite...${NC}"
$ARCH_CMD python3 -m pytest tests/ -q

echo -e "\n${GREEN}[2/4] Testing Pinecone CLI (pc) status...${NC}"
if command -v pc &> /dev/null; then
    pc version
else
    echo "Pinecone CLI not in PATH."
fi

echo -e "\n${GREEN}[3/4] Populating Vector Index & Running Benchmark Queries...${NC}"
$ARCH_CMD python3 -m src.cli.main demo

echo -e "\n${GREEN}[4/4] Starting Glassmorphism Dashboard Server...${NC}"
echo -e "Dashboard active at: ${CYAN}http://localhost:8000${NC}"
echo -e "Press Ctrl+C to stop server.\n"

$ARCH_CMD python3 -m src.web.app
