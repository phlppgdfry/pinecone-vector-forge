# Pinecone CLI (`pc`) Production Cheat Sheet

A comprehensive, battle-tested operational guide for managing vector infrastructure directly from your command line using the official **Pinecone CLI** (`pc`).

---

## 1. Installation & Environment Verification

### macOS (Homebrew Cask)
```bash
brew tap pinecone-io/tap
arch -arm64 brew install --cask pinecone-io/tap/pinecone
```

### Verify Version & Binary Location
```bash
pc version
which pc
```

---

## 2. Authentication & Credential Management

### Interactive Browser SSO Login
```bash
# Opens browser to authenticate your Pinecone account
pc login
```

### Inspect Active Session Status
```bash
# Check logged-in user, active API key, and environment
pc auth status
pc whoami
```

### Logout & Wipe Local Credentials
```bash
pc logout
```

---

## 3. Serverless Index Lifecycle Management

### List All Indexes in Project
```bash
pc index list
```

### Create a Serverless Index (AWS us-east-1)
```bash
pc index create \
  --name vector-forge-knowledge \
  --dimension 1536 \
  --metric cosine \
  --spec-cloud aws \
  --spec-region us-east-1
```

### Inspect Index Telemetry & Endpoints
```bash
pc index describe --name vector-forge-knowledge
```

### Delete an Index
```bash
pc index delete --name vector-forge-knowledge --confirm
```

---

## 4. Namespace Partitioning

### List Namespaces inside an Index
```bash
pc index namespace list --index vector-forge-knowledge
```

### Delete a Specific Namespace (Tenant Offboarding / GDPR)
```bash
pc index namespace delete \
  --index vector-forge-knowledge \
  --namespace tenant_acme_corp
```

---

## 5. Record Querying & Debugging from Terminal

### Search Nearest Neighbors Directly
```bash
pc index record search \
  --index vector-forge-knowledge \
  --namespace default \
  --top-k 5
```

---

## 6. GitOps & CI/CD Integration Example

```yaml
# In GitHub Actions
- name: Install Pinecone CLI
  run: |
    brew tap pinecone-io/tap
    brew install --cask pinecone-io/tap/pinecone

- name: Audit Vector Index Health
  env:
    PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
  run: |
    pc index describe --name vector-forge-knowledge
```
