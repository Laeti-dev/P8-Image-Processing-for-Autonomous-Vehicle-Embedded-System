# Azure Guide - Free Deployment for Students

## 🎓 Azure for Students Account

### Account Creation
1. Go to: https://azure.microsoft.com/en-us/free/students/
2. Verify eligibility (student email required)
3. Create account ($100 free credits)
4. **Important**: Credits expire after 12 months

### Available Free Services

#### Azure App Service (Recommended for this project)
- **F1 Tier (Free)**:
  - 1 GB storage
  - 60 minutes CPU/day
  - 1 GB bandwidth/day
  - **Limitation**: App sleeps after 20 minutes of inactivity
  - **Solution**: Use a free "ping" service (UptimeRobot, etc.)

#### Azure Blob Storage
- **Standard LRS Tier**:
  - 5 GB free storage
  - Ideal for storing trained models
  - 20,000 free reads/day

#### Azure Container Instances
- **Alternative option** if App Service is not sufficient
- More complex to configure

---

## 🚀 API Deployment on Azure App Service

### Option 1: Deployment via Azure CLI (Recommended)

#### Prerequisites
```bash
# Install Azure CLI
# macOS
brew install azure-cli

# Login
az login

# Verify connection
az account show
```

#### App Service Creation
```bash
# First, check available locations for your subscription
# Some subscriptions have region restrictions
az account list-locations --query "[?metadata.regionCategory=='Recommended'].{Name:name, DisplayName:displayName}" --output table

# Variables
RESOURCE_GROUP="rg-segmentation-project"
APP_NAME="api-segmentation-$(date +%s)"  # Unique name
# Try these regions in order if one fails:
# - germanywestcentral (Germany West Central) - commonly available for students
# - eastus (East US) - most commonly available
# - northeurope (North Europe)
# - westeurope (West Europe) - may be restricted
# - uksouth (UK South)
LOCATION="germanywestcentral"  # Change if you get region restriction error

# Create Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan (free)
az appservice plan create \
  --name plan-segmentation-free \
  --resource-group $RESOURCE_GROUP \
  --sku FREE \
  --location $LOCATION

# Create App Service
az webapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan plan-segmentation-free \
  --runtime "PYTHON:3.9"

# Configure startup
az webapp config set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --startup-file "gunicorn api.app:app --bind 0.0.0.0:8000"
```

#### FastAPI Configuration
```bash
# Install gunicorn in requirements.txt
# gunicorn
# uvicorn[standard]

# Configure port (Azure uses PORT env variable)
# In api/app.py:
# port = int(os.environ.get("PORT", 8000))
```

#### Code Deployment
```bash
# Option A: Local deployment
cd api/
az webapp up \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --runtime "PYTHON:3.9"

# Option B: Git deployment
az webapp deployment source config-local-git \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP
```

### Option 2: Deployment via GitHub Actions (Automated)

#### Create workflow `.github/workflows/deploy-api.yml`
```yaml
name: Deploy API to Azure

on:
  push:
    branches: [ main ]
    paths:
      - 'api/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Deploy to Azure Web App
        uses: azure/webapps-deploy@v2
        with:
          app-name: 'api-segmentation'
          package: './api'
```

### Option 3: Deployment via VS Code Extension

1. Install "Azure App Service" extension
2. Connect to Azure
3. Right-click on `api/` folder → "Deploy to Web App"

---

## 🌐 Web Application Deployment

### Same process as API
```bash
# Create a second App Service
WEB_APP_NAME="web-segmentation-$(date +%s)"

az webapp create \
  --name $WEB_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan plan-segmentation-free \
  --runtime "PYTHON:3.9"
```

### Streamlit Configuration on Azure
```python
# In web_app/app.py, add at end of file:
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    streamlit.run(port=port)
```

**Note**: Streamlit requires special configuration. Alternative: use Flask which is simpler on Azure.

---

## 💾 Model Storage

### Option 1: Include in Deployment
- Limit: 1 GB on F1 tier
- Simple but less flexible

### Option 2: Azure Blob Storage (Recommended)

#### Create Storage Account
```bash
STORAGE_ACCOUNT="storage$(date +%s)"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Create container
az storage container create \
  --name models \
  --account-name $STORAGE_ACCOUNT \
  --public-access blob
```

#### Upload Model
```bash
# Install Azure Storage Blob library
pip install azure-storage-blob

# Upload
az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --container-name models \
  --name model.h5 \
  --file models/model.h5 \
  --overwrite
```

#### Load in API
```python
# In api/model_loader.py
from azure.storage.blob import BlobServiceClient

def load_model_from_blob():
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container="models", blob="model.h5")

    # Download and load model
    # ...
```

---

## ⚙️ Configuration and Environment Variables

### Set Environment Variables
```bash
# Via Azure CLI
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    API_URL="https://api-segmentation.azurewebsites.net" \
    MODEL_PATH="/models/model.h5"
```

### Access in Code
```python
import os

api_url = os.environ.get("API_URL", "http://localhost:8000")
model_path = os.environ.get("MODEL_PATH", "./models/model.h5")
```

---

## 🔍 Monitoring and Logs

### View Logs
```bash
# Real-time logs
az webapp log tail \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP

# Download logs
az webapp log download \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --log-file logs.zip
```

### Monitoring in Azure Portal
- Go to Azure Portal
- Select App Service
- "Monitoring" section → "Log stream"

---

## 💰 Cost Management

### Monitor Usage
```bash
# View current usage
az consumption usage list \
  --start-date $(date -d "1 month ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d)
```

### Tips to Stay Free
1. **Use only F1 tier** (Free)
2. **Limit model size** (< 500 MB recommended)
3. **Optimize requests** (caching, compression)
4. **Disable unused apps**
5. **Regularly monitor** usage

### Cost Alerts
1. Go to "Cost Management + Billing"
2. Create a budget
3. Configure alerts (e.g., alert at $5)

---

## 🐛 Common Issues and Solutions

### Issue: Region Not Allowed / RequestDisallowedByAzure
**Error**: `Resource was disallowed by Azure: This policy maintains a set of best available regions...`

**Solution**: Your subscription has region restrictions. Try these steps:

1. **Check available regions for your subscription**:
```bash
az account list-locations --query "[?metadata.regionCategory=='Recommended'].{Name:name, DisplayName:displayName}" --output table
```

2. **Try a different region** (most commonly available):
```bash
# Change LOCATION variable to one of these:
LOCATION="germanywestcentral"  # Germany West Central (commonly available for students)
# or
LOCATION="eastus"              # East US (most commonly available)
# or
LOCATION="northeurope"         # North Europe
# or
LOCATION="uksouth"             # UK South
```

3. **If still failing, check subscription policies**:
```bash
az account show --query "{SubscriptionId:id, Name:name}"
# Then check in Azure Portal: Subscriptions → Your Subscription → Policies
```

4. **For Azure for Students accounts**, some regions may be restricted. Contact Azure support if needed.

### Issue: App sleeps after 20 minutes
**Solution**: Use a free ping service
- UptimeRobot (free up to 50 monitors)
- Configure ping every 5 minutes to `/health` endpoint

### Issue: "Module not found" Error
**Solution**: Check `requirements.txt`
```bash
# Manually install dependencies
az webapp ssh \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP

# Then in shell:
pip install -r requirements.txt
```

### Issue: Port Already in Use
**Solution**: Use PORT environment variable
```python
port = int(os.environ.get("PORT", 8000))
```

### Issue: Model Too Heavy
**Solution**:
- Use Azure Blob Storage
- Or optimize model (quantization, pruning)

---

## 📚 Useful Resources

- Azure App Service Documentation: https://docs.microsoft.com/en-us/azure/app-service/
- Pricing Calculator: https://azure.microsoft.com/en-us/pricing/calculator/
- Azure CLI Reference: https://docs.microsoft.com/en-us/cli/azure/
- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/

---

## ✅ Azure Deployment Checklist

- [ ] Azure for Students account created
- [ ] Azure CLI installed and configured
- [ ] Resource Group created
- [ ] App Service Plan (F1) created
- [ ] API App Service created
- [ ] Web App Service created
- [ ] Model uploaded (Blob Storage or in app)
- [ ] Environment variables configured
- [ ] API tested and functional
- [ ] Web application tested and functional
- [ ] Ping service configured (to avoid sleep)
- [ ] Logs verified
- [ ] Monitoring configured

---

**Important Note**: The free F1 tier has limitations. For a demonstration project, it's sufficient, but for production, you'll need to upgrade to a paid tier.
