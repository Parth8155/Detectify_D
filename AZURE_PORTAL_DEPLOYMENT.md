# 🌐 Azure Portal GUI Deployment Guide for Detectify
*Complete Step-by-Step Guide Using Azure Web Portal*

## 📋 **Prerequisites**
- Azure Account (free tier works)
- Your Detectify project pushed to GitHub
- Basic understanding of Azure Portal

---

## 🚀 **STEP 1: Create Resource Group**

### 1.1 Login to Azure Portal
1. Go to [https://portal.azure.com](https://portal.azure.com)
2. Sign in with your Azure account
3. Click on **"Resource groups"** in the left sidebar

### 1.2 Create New Resource Group
1. Click **"+ Create"** button
2. Fill in the details:
   - **Subscription**: Select your subscription
   - **Resource group name**: `detectify-rg`
   - **Region**: Choose closest to you (e.g., "East US", "West Europe")
3. Click **"Review + create"**
4. Click **"Create"**

---

## 🗄️ **STEP 2: Create PostgreSQL Database**

### 2.1 Navigate to Database Creation
1. In Azure Portal, click **"+ Create a resource"**
2. Search for **"Azure Database for PostgreSQL"**
3. Click **"Azure Database for PostgreSQL servers"**
4. Click **"Create"**

### 2.2 Configure PostgreSQL Server
1. **Basics Tab:**
   - **Subscription**: Your subscription
   - **Resource group**: `detectify-rg`
   - **Server name**: `detectify-db-server` (must be unique)
   - **Data source**: None (blank database)
   - **Location**: Same as resource group
   - **Version**: 13 or 14
   - **Compute + storage**: Click "Configure server"
     - Choose **"Basic"** tier (for cost-effective)
     - **vCores**: 1 or 2
     - **Storage**: 32 GB
     - **Backup retention**: 7 days
   - **Admin username**: `detectify_admin`
   - **Password**: Create a strong password (save it!)

2. **Networking Tab:**
   - **Connectivity method**: Public access
   - **Allow access to Azure services**: YES
   - **Add current client IP**: YES (for management)

3. Click **"Review + create"**
4. Click **"Create"** (takes 5-10 minutes)

### 2.3 Create Database
1. Once PostgreSQL server is created, go to the resource
2. Click **"Databases"** in left sidebar
3. Click **"+ Add"**
4. **Database name**: `detectify`
5. Click **"Save"**

### 2.4 Configure Firewall
1. In PostgreSQL server, click **"Connection security"**
2. Set **"Allow access to Azure services"**: ON
3. Add firewall rule:
   - **Rule name**: `AllowAllAzure`
   - **Start IP**: `0.0.0.0`
   - **End IP**: `0.0.0.0`
4. Click **"Save"**

---

## 🔄 **STEP 3: Create Redis Cache (Optional but Recommended)**

### 3.1 Create Redis Cache
1. Click **"+ Create a resource"**
2. Search for **"Azure Cache for Redis"**
3. Click **"Create"**

### 3.2 Configure Redis
1. **Basics:**
   - **Subscription**: Your subscription
   - **Resource group**: `detectify-rg`
   - **DNS name**: `detectify-redis` (must be unique)
   - **Location**: Same region
   - **Cache type**: Basic C0 (250 MB) - cheapest option

2. **Networking:** Leave default (Public endpoint)
3. **Advanced:** Leave defaults
4. Click **"Review + create"**
5. Click **"Create"** (takes 10-15 minutes)

---

## 🌐 **STEP 4: Create App Service**

### 4.1 Create App Service Plan
1. Click **"+ Create a resource"**
2. Search for **"App Service"**
3. Click **"Web App"** and then **"Create"**

### 4.2 Configure Web App
1. **Basics Tab:**
   - **Subscription**: Your subscription
   - **Resource group**: `detectify-rg`
   - **Name**: `detectify-app-[your-unique-id]` (must be globally unique)
   - **Publish**: Code
   - **Runtime stack**: Python 3.11
   - **Operating System**: Linux
   - **Region**: Same as your resource group

2. **App Service Plan:**
   - Click **"Create new"**
   - **Name**: `detectify-plan`
   - **Pricing tier**: 
     - For testing: **F1 (Free)**
     - For production: **B1 (Basic)** or higher

3. **Monitoring Tab:**
   - **Enable Application Insights**: Yes (recommended)
   - **Application Insights**: Create new or use existing

4. Click **"Review + create"**
5. Click **"Create"**

---

## 📂 **STEP 5: Deploy Your Code**

### 5.1 Connect to GitHub
1. Go to your newly created App Service
2. In left sidebar, click **"Deployment Center"**
3. **Source**: Select **"GitHub"**
4. Click **"Authorize"** and sign in to GitHub
5. **Organization**: Your GitHub username
6. **Repository**: `Detectify_D` (your repo name)
7. **Branch**: `master`
8. Click **"Save"**

### 5.2 Configure Build Settings
1. **Build provider**: App Service build service
2. **Runtime stack**: Python 3.11
3. **Runtime version**: 3.11
4. **Startup command**: 
   ```bash
   python -m gunicorn detectify_project.wsgi:application --bind 0.0.0.0:8000 --timeout 120
   ```
5. Click **"Save"**

---

## ⚙️ **STEP 6: Configure Environment Variables**

### 6.1 Add Application Settings
1. In your App Service, go to **"Configuration"** in left sidebar
2. Click **"+ New application setting"** for each variable below:

### 6.2 Required Environment Variables

#### Django Settings:
```
Name: DJANGO_SETTINGS_MODULE
Value: detectify_project.settings.production
```

```
Name: SECRET_KEY
Value: [Generate a new secret key - use Django's get_random_secret_key()]
```

#### Database Settings:
```
Name: AZURE_POSTGRESQL_NAME
Value: detectify
```

```
Name: AZURE_POSTGRESQL_USER
Value: detectify_admin@detectify-db-server
```

```
Name: AZURE_POSTGRESQL_PASSWORD
Value: [Your PostgreSQL password]
```

```
Name: AZURE_POSTGRESQL_HOST
Value: detectify-db-server.postgres.database.azure.com
```

```
Name: AZURE_POSTGRESQL_PORT
Value: 5432
```

#### Redis Settings (if you created Redis):
```
Name: REDIS_URL
Value: redis://detectify-redis.redis.cache.windows.net:6380?ssl_cert_reqs=required&password=[Redis_Access_Key]
```

*To get Redis access key:*
1. Go to your Redis cache resource
2. Click "Access keys" in left sidebar
3. Copy the "Primary connection string"

#### Website Hostname (Auto-set by Azure):
```
Name: WEBSITE_HOSTNAME
Value: [This is automatically set by Azure - you don't need to add this]
```

### 6.3 Save Configuration
1. Click **"Save"** after adding all variables
2. Click **"Continue"** when prompted about restart

---

## 🔧 **STEP 7: Configure Startup Script**

### 7.1 Set Startup Command
1. In App Service **"Configuration"**
2. Go to **"General settings"** tab
3. **Startup Command**: 
   ```bash
   bash startup.sh
   ```
4. **Python version**: 3.11
5. Click **"Save"**

---

## 🚀 **STEP 8: Initial Deployment & Setup**

### 8.1 Trigger Deployment
1. Go to **"Deployment Center"**
2. Click **"Sync"** to trigger a fresh deployment
3. Monitor the **"Logs"** tab to see deployment progress

### 8.2 Run Database Migrations
1. Go to **"Development Tools"** → **"SSH"**
2. Click **"Go"** to open web SSH terminal
3. Run these commands:
   ```bash
   cd /home/site/wwwroot
   python manage.py migrate
   python manage.py collectstatic --noinput
   python manage.py create_azure_superuser
   ```

---

## 🔍 **STEP 9: Test Your Deployment**

### 9.1 Access Your Application
1. In App Service **"Overview"**, find the **URL**
2. Click the URL to open your app
3. You should see your Detectify homepage

### 9.2 Test Admin Access
1. Go to `https://your-app-name.azurewebsites.net/admin/`
2. Login with:
   - Username: `admin`
   - Password: `admin123` (change this immediately!)

### 9.3 Create Real Admin User
1. In SSH terminal:
   ```bash
   python manage.py createsuperuser
   ```
2. Follow prompts to create a secure admin user

---

## 🔧 **STEP 10: Configure Custom Domain (Optional)**

### 10.1 Add Custom Domain
1. In App Service, go to **"Custom domains"**
2. Click **"+ Add custom domain"**
3. Enter your domain name
4. Follow verification steps
5. Configure DNS records as instructed

### 10.2 Add SSL Certificate
1. Go to **"TLS/SSL settings"**
2. Click **"+ Add TLS/SSL binding"**
3. Select your custom domain
4. Choose **"SNI SSL"**
5. Select or create SSL certificate

---

## 📊 **STEP 11: Monitor Your Application**

### 11.1 Application Insights
1. Go to **"Application Insights"** from your App Service
2. Monitor performance, errors, and usage
3. Set up alerts for critical issues

### 11.2 Log Monitoring
1. Go to **"App Service logs"** in your App Service
2. Enable **"Application logging"**
3. Set level to **"Information"** or **"Warning"**
4. View logs in **"Log stream"**

---

## 🛠️ **STEP 12: Production Optimizations**

### 12.1 Scale Up Your App Service
1. Go to **"Scale up (App Service plan)"**
2. Choose a higher tier (B1, S1, P1v2, etc.)
3. Consider auto-scaling for variable load

### 12.2 Configure Backup (Recommended)
1. Go to **"Backups"**
2. Configure automated backups
3. Include database in backup

### 12.3 Set Up Deployment Slots (Advanced)
1. Go to **"Deployment slots"**
2. Add **"Staging"** slot
3. Use for testing before production deployment

---

## ❌ **Troubleshooting Common Issues**

### Issue 1: "Application Error"
**Solution:**
1. Check logs in **"Log stream"**
2. Verify environment variables
3. Ensure database connection works

### Issue 2: Static Files Not Loading
**Solution:**
1. Run `python manage.py collectstatic --noinput` via SSH
2. Verify `STATIC_ROOT` setting
3. Check if WhiteNoise is properly configured

### Issue 3: Database Connection Errors
**Solution:**
1. Verify PostgreSQL firewall rules
2. Check connection string format
3. Ensure SSL is enabled

### Issue 4: Redis Connection Issues
**Solution:**
1. Verify Redis access keys
2. Check SSL requirements in connection string
3. Test Redis connectivity

---

## 💰 **Cost Optimization Tips**

1. **Use Free Tier**: Start with F1 for testing
2. **Monitor Usage**: Use Azure Cost Management
3. **Scale Down**: Reduce resources during low usage
4. **Reserved Instances**: Consider for long-term usage
5. **Auto-shutdown**: Configure for dev/test environments

---

## 🔐 **Security Best Practices**

1. **Change Default Passwords**: Update admin credentials
2. **Enable HTTPS**: Force HTTPS redirects
3. **Regular Updates**: Keep dependencies updated
4. **Backup Strategy**: Regular database backups
5. **Monitor Logs**: Watch for suspicious activity
6. **Access Controls**: Limit admin access

---

## 🎉 **Congratulations!**

Your Detectify application is now live on Azure! 

**Your app URL:** `https://your-app-name.azurewebsites.net`

### Next Steps:
1. Test all functionality thoroughly
2. Upload some test images and videos
3. Configure email notifications (optional)
4. Set up monitoring alerts
5. Plan your backup strategy
6. Consider CDN for better performance

### Support Resources:
- Azure Documentation: [https://docs.microsoft.com/azure](https://docs.microsoft.com/azure)
- Django on Azure: [https://docs.microsoft.com/azure/app-service/configure-language-python](https://docs.microsoft.com/azure/app-service/configure-language-python)
- Azure Support: Available through Azure Portal

---

*Happy face detecting in the cloud! 🔍☁️*
