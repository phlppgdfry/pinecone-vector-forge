#!/usr/bin/env bash
# ==============================================================================
# Pinecone Vector Forge - Push Knowledge Corpus to Live Pinecone Dashboard
# ==============================================================================
set -e

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}   🌲 Pinecone Vector Forge — Cloud Dashboard Synchronization     ${NC}"
echo -e "${CYAN}=================================================================${NC}"

API_KEY="${1:-$PINECONE_API_KEY}"

# Prompt for key if not passed or in env
if [ -z "$API_KEY" ] || [ "$API_KEY" = "your_pinecone_api_key_here" ]; then
    echo -e "${YELLOW}No PINECONE_API_KEY found.${NC}"
    echo -n "Paste your Pinecone API Key (pcsk_...): "
    read -r API_KEY
fi

if [ -z "$API_KEY" ]; then
    echo -e "${RED}Error: API key is required to sync with Pinecone Cloud.${NC}"
    echo "Get your key for free at https://app.pinecone.io"
    exit 1
fi

echo -e "\n${GREEN}[1/4] Configuring Pinecone CLI credentials...${NC}"
pc auth configure --api-key "$API_KEY"

echo -e "\n${GREEN}[2/4] Verifying Pinecone Cloud Authentication...${NC}"
pc auth status

INDEX_NAME="vector-forge-knowledge"
echo -e "\n${GREEN}[3/4] Ensuring Serverless Index '${INDEX_NAME}' exists (AWS us-east-1)...${NC}"
if ! pc index list | grep -q "$INDEX_NAME"; then
    echo "Creating serverless index '$INDEX_NAME'..."
    pc index create \
      --name "$INDEX_NAME" \
      --dimension 1536 \
      --metric cosine \
      --cloud aws \
      --region us-east-1
    echo "Index created. Waiting 5 seconds for cloud DNS propagation..."
    sleep 5
else
    echo "Index '$INDEX_NAME' already exists in your Pinecone project."
fi

echo -e "\n${GREEN}[4/4] Upserting Vector Knowledge Corpus to Cloud...${NC}"
export PINECONE_API_KEY="$API_KEY"
arch -arm64 python3 -m src.seed.ingest

echo -e "\n${CYAN}=================================================================${NC}"
echo -e "${GREEN}✔ Successfully pushed to your Pinecone Cloud Dashboard!${NC}"
echo -e "View your live vectors at: ${CYAN}https://app.pinecone.io${NC}"
echo -e "${CYAN}=================================================================${NC}\n"
