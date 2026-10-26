#!/usr/bin/env bash
# ==============================================================================
# Pinecone Vector Forge - Pinecone CLI ('pc') Quick Helper
# ==============================================================================
set -e

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
NC="\033[0m"

echo -e "${CYAN}🌲 Pinecone CLI Assistant (pc helper)${NC}"

if ! command -v pc &> /dev/null; then
    echo -e "${YELLOW}Warning: 'pc' command not found in PATH.${NC}"
    echo "Install via Homebrew: arch -arm64 brew install --cask pinecone-io/tap/pinecone"
    exit 1
fi

ACTION=${1:-"help"}

case "$ACTION" in
    "login")
        echo -e "${GREEN}Initiating official Pinecone browser login...${NC}"
        pc login
        ;;
    "status"|"whoami")
        echo -e "${GREEN}Checking active Pinecone auth status...${NC}"
        pc auth status
        ;;
    "indexes"|"list")
        echo -e "${GREEN}Listing your Pinecone indexes...${NC}"
        pc index list
        ;;
    "create")
        INDEX_NAME=${2:-"vector-forge-knowledge"}
        echo -e "${GREEN}Provisioning Serverless Index: ${INDEX_NAME} (AWS / us-east-1)...${NC}"
        pc index create --name "$INDEX_NAME" --dimension 1536 --metric cosine --spec-cloud aws --spec-region us-east-1
        ;;
    "describe")
        INDEX_NAME=${2:-"vector-forge-knowledge"}
        echo -e "${GREEN}Describing index: ${INDEX_NAME}...${NC}"
        pc index describe --name "$INDEX_NAME"
        ;;
    *)
        echo "Usage: $0 {login|status|indexes|create [index-name]|describe [index-name]}"
        echo ""
        echo "Commands:"
        echo "  login     - Log in to your Pinecone account via browser SSO"
        echo "  status    - View current API key and authentication state"
        echo "  indexes   - List all indexes in your active Pinecone project"
        echo "  create    - Create a Serverless 1536-d index (AWS us-east-1)"
        echo "  describe  - Show index statistics and endpoint details"
        ;;
esac
