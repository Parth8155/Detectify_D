# 🚀 Detectify Azure Deployment Guide - A to Z

## 📋 Prerequisites

- Azure account with active subscription
- Azure CLI installed on your machine
- Git repository with your code
- Basic knowledge of Azure services

## 🏗️ Azure Services Required

1. **Azure App Service** - Web application hosting
2. **Azure Database for PostgreSQL** - Production database
3. **Azure Cache for Redis** - Caching and Celery backend
4. **Azure Storage Account** - Media files storage (optional)
5. **Azure Application Insights** - Monitoring (optional)

---

## 📦 Step 1: Prepare Your Local Environment

### 1.1 Install Azure CLI
```bash
# Windows
winget install Microsoft.AzureCLI

# Or download from: https://aka.ms/installazurecliwindows
```

### 1.2 Login to Azure
```bash
az login
```

### 1.3 Set Your Subscription
```bash
# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "your-subscription-id"
```

---

## 🗄️ Step 2: Create Azure Database for PostgreSQL

### 2.1 Create Resource Group
```bash
az group create --name detectify-rg --location "East US"
```

### 2.2 Create PostgreSQL Server
```bash
az postgres server create \
  --resource-group detectify-rg \
  --name detectify-db-server \
  --location "East US" \
  --admin-user detectify_admin \
  --admin-password "YourSecurePassword123!" \
  --sku-name GP_Gen5_2 \
  --version 13
```

### 2.3 Create Database
```bash
az postgres db create \
  --resource-group detectify-rg \
  --server-name detectify-db-server \
  --name detectify_db
```

### 2.4 Configure Firewall Rules
```bash
# Allow Azure services
az postgres server firewall-rule create \
  --resource-group detectify-rg \
  --server detectify-db-server \
  --name "AllowAzureServices" \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Allow your IP (for testing)
az postgres server firewall-rule create \
  --resource-group detectify-rg \
  --server detectify-db-server \
  --name "AllowMyIP" \
  --start-ip-address YOUR_IP \
  --end-ip-address YOUR_IP
```

---

## 🔄 Step 3: Create Azure Cache for Redis

```bash
az redis create \
  --resource-group detectify-rg \
  --name detectify-redis \
  --location "East US" \
  --sku Basic \
  --vm-size c0
```

---

## 📁 Step 4: Create Azure Storage Account (Optional)

```bash
# Create storage account
az storage account create \
  --name detectifystorage \
  --resource-group detectify-rg \
  --location "East US" \
  --sku Standard_LRS

# Create container for media files
az storage container create \
  --name media \
  --account-name detectifystorage \
  --public-access blob
```

---

## 🌐 Step 5: Create Azure App Service

### 5.1 Create App Service Plan
```bash
az appservice plan create \
  --name detectify-plan \
  --resource-group detectify-rg \
  --location "East US" \
  --sku B1 \
  --is-linux
```

### 5.2 Create Web App
```bash
az webapp create \
  --resource-group detectify-rg \
  --plan detectify-plan \
  --name detectify-app-unique \
  --runtime "PYTHON|3.11" \
  --deployment-source-url https://github.com/YourUsername/YourRepo.git \
  --deployment-source-branch master
```

---

## ⚙️ Step 6: Configure Application Settings

### 6.1 Get Connection Strings
```bash
# Get PostgreSQL connection string
az postgres server show-connection-string \
  --server-name detectify-db-server \
  --database-name detectify_db \
  --admin-user detectify_admin \
  --admin-password "YourSecurePassword123!"

# Get Redis connection string
az redis list-keys \
  --resource-group detectify-rg \
  --name detectify-redis
```

### 6.2 Set Environment Variables
```bash
# Set Django settings
az webapp config appsettings set \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --settings \
  DJANGO_SETTINGS_MODULE=detectify_project.settings.production \
  SECRET_KEY="your-super-secret-key-here" \
  AZURE_POSTGRESQL_NAME=detectify_db \
  AZURE_POSTGRESQL_USER=detectify_admin \
  AZURE_POSTGRESQL_PASSWORD="YourSecurePassword123!" \
  AZURE_POSTGRESQL_HOST=detectify-db-server.postgres.database.azure.com \
  AZURE_POSTGRESQL_PORT=5432 \
  REDIS_URL="redis://:your-redis-key@detectify-redis.redis.cache.windows.net:6380?ssl_cert_reqs=required"
```

### 6.3 Configure Startup Command
```bash
az webapp config set \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --startup-file "startup.sh"
```

---

## 📤 Step 7: Deploy Your Code

### Option 1: GitHub Deployment (Recommended)

```bash
# Configure GitHub deployment
az webapp deployment source config \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --repo-url https://github.com/YourUsername/YourRepo \
  --branch master \
  --manual-integration
```

### Option 2: Local Git Deployment

```bash
# Get deployment credentials
az webapp deployment user set \
  --user-name your-username \
  --password your-password

# Get Git URL
az webapp deployment source config-local-git \
  --resource-group detectify-rg \
  --name detectify-app-unique

# Add Azure remote and deploy
git remote add azure https://your-username@detectify-app-unique.scm.azurewebsites.net/detectify-app-unique.git
git push azure master
```

### Option 3: ZIP Deployment

```bash
# Create deployment package
zip -r detectify.zip . -x "*.git*" "__pycache__/*" "*.pyc" "venv/*"

# Deploy ZIP
az webapp deployment source config-zip \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --src detectify.zip
```

---

## 🔧 Step 8: Post-Deployment Configuration

### 8.1 Run Database Migrations
```bash
# SSH into your app (if needed)
az webapp ssh \
  --resource-group detectify-rg \
  --name detectify-app-unique

# Or use the web SSH console in Azure portal
# Navigate to: App Service → Development Tools → SSH
```

### 8.2 Create Superuser
```bash
# In SSH console or using Azure Cloud Shell
python manage.py create_azure_superuser
```

### 8.3 Collect Static Files
```bash
python manage.py collectstatic --noinput
```

---

## 🔒 Step 9: Security Configuration

### 9.1 Configure Custom Domain (Optional)
```bash
# Add custom domain
az webapp config hostname add \
  --resource-group detectify-rg \
  --webapp-name detectify-app-unique \
  --hostname yourdomain.com

# Add SSL certificate
az webapp config ssl upload \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --certificate-file path/to/certificate.pfx \
  --certificate-password certificate-password
```

### 9.2 Configure Authentication (Optional)
```bash
# Enable Azure AD authentication
az webapp auth update \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --enabled true \
  --action LoginWithAzureActiveDirectory
```

---

## 📊 Step 10: Monitoring and Logging

### 10.1 Enable Application Insights
```bash
# Create Application Insights
az monitor app-insights component create \
  --app detectify-insights \
  --location "East US" \
  --resource-group detectify-rg \
  --application-type web

# Link to App Service
az webapp config appsettings set \
  --resource-group detectify-rg \
  --name detectify-app-unique \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY="your-insights-key"
```

### 10.2 Configure Log Stream
```bash
# View live logs
az webapp log tail \
  --resource-group detectify-rg \
  --name detectify-app-unique
```

---

## 🧪 Step 11: Testing Your Deployment

### 11.1 Basic Health Check
1. Visit your app URL: `https://detectify-app-unique.azurewebsites.net`
2. Test login functionality
3. Test case creation
4. Check admin panel: `https://your-app.azurewebsites.net/admin/`

### 11.2 Performance Testing
```bash
# Use Azure Load Testing or tools like ApacheBench
ab -n 100 -c 10 https://your-app.azurewebsites.net/
```

---

## 🔄 Step 12: CI/CD Pipeline (Optional)

### Using GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure App Service

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python manage.py test
        
    - name: Deploy to Azure
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'detectify-app-unique'
        slot-name: 'production'
        publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
```

---

## 📋 Step 13: Production Checklist

### ✅ Security
- [ ] SECRET_KEY is secure and unique
- [ ] DEBUG = False
- [ ] ALLOWED_HOSTS configured correctly
- [ ] SSL certificate installed
- [ ] Database firewall rules configured
- [ ] Redis password protected

### ✅ Performance
- [ ] Static files served by CDN or WhiteNoise
- [ ] Database connection pooling enabled
- [ ] Redis caching configured
- [ ] Application Insights monitoring

### ✅ Functionality
- [ ] Database migrations completed
- [ ] Superuser account created
- [ ] Media files upload working
- [ ] Email configuration tested
- [ ] Celery tasks working (if used)

---

## 🚨 Common Issues and Solutions

### Issue 1: Static Files Not Loading
**Solution:**
```bash
# Ensure STATIC_ROOT is set correctly
python manage.py collectstatic --noinput

# Check WhiteNoise configuration in settings
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
```

### Issue 2: Database Connection Failed
**Solution:**
- Check firewall rules
- Verify connection string
- Ensure SSL is enabled

### Issue 3: Application Not Starting
**Solution:**
```bash
# Check logs
az webapp log tail --resource-group detectify-rg --name detectify-app-unique

# Verify startup.sh is executable
chmod +x startup.sh
```

### Issue 4: 500 Internal Server Error
**Solution:**
- Check Application Insights or logs
- Verify all environment variables
- Test database connectivity

---

## 💰 Cost Optimization Tips

1. **Use appropriate SKUs:**
   - App Service: Start with B1, scale as needed
   - Database: Use GP_Gen5_1 for development
   - Redis: Basic tier for small apps

2. **Monitor usage:**
   - Set up billing alerts
   - Use Azure Cost Management
   - Scale down non-production environments

3. **Optimize resources:**
   - Use connection pooling
   - Implement caching
   - Optimize database queries

---

## 📞 Support and Resources

- **Azure Documentation:** [docs.microsoft.com/azure](https://docs.microsoft.com/azure)
- **Django on Azure:** [docs.microsoft.com/azure/app-service/quickstart-python](https://docs.microsoft.com/azure/app-service/quickstart-python)
- **Azure Support:** [azure.microsoft.com/support](https://azure.microsoft.com/support)

---

## 🎉 Congratulations!

Your Detectify application is now deployed on Azure! Remember to:

1. Monitor your application regularly
2. Keep dependencies updated
3. Backup your database
4. Scale resources as needed
5. Implement proper logging and monitoring

Your app should be accessible at: `https://detectify-app-unique.azurewebsites.net`

Happy detecting! 🔍✨
