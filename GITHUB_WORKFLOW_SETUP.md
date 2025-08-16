# 🔄 GitHub Actions Workflow Setup for Azure Deployment

## 📋 **Workflow Options Explained**

When deploying to Azure, you have these workflow options:

### ✅ **Option 1: Use Available Workflow (RECOMMENDED)**
- **What it means**: Use the existing `master_Detectify.yml` file in your repo
- **Best for**: Full control and customization
- **Setup time**: 10-15 minutes
- **Authentication**: Publish Profile (easier) or Service Principal (advanced)

### 🆕 **Option 2: Add a Workflow**
- **What it means**: Let Azure generate a new workflow file automatically
- **Best for**: Quick setup with minimal configuration
- **Setup time**: 5 minutes
- **Authentication**: Usually Publish Profile

---

## 🚀 **RECOMMENDED: Use Available Workflow**

I've already created the perfect workflow file for you. Here's what to do:

### **Step 1: Choose Your Authentication Method**

#### **Method A: Publish Profile (EASIER - Recommended for beginners)**
- ✅ Simple one-click setup
- ✅ No Azure CLI required
- ✅ Works immediately
- ❌ Less secure than Service Principal

#### **Method B: Service Principal (More Secure)**
- ✅ More secure
- ✅ Better for production
- ✅ Fine-grained permissions
- ❌ Requires Azure CLI setup

---

## 🔧 **Setup Guide for Method A (Publish Profile)**

### **Step 1: Get Publish Profile from Azure Portal**

1. **Go to your Azure App Service** in Azure Portal
2. **Click "Get publish profile"** button (top toolbar)
3. **Download the `.publishsettings` file**
4. **Open the file** in text editor and **copy all contents**

### **Step 2: Add Secret to GitHub**

1. **Go to your GitHub repository**
2. **Settings** → **Secrets and variables** → **Actions**
3. **Click "New repository secret"**
4. **Name**: `AZURE_WEBAPP_PUBLISH_PROFILE`
5. **Value**: Paste the entire publish profile content
6. **Click "Add secret"**

### **Step 3: Update Workflow File**

Use the simple workflow I created (`simple_azure_deploy.yml`) or update the main workflow:

```yaml
# In your workflow, replace this line:
app-name: 'your-app-name'  
# With your actual Azure app name:
app-name: 'detectify-app-your-unique-id'
```

### **Step 4: Push Changes and Deploy**

```bash
git add .
git commit -m "Add Azure deployment workflow"
git push origin master
```

The deployment will start automatically!

---

## 🔒 **Setup Guide for Method B (Service Principal)**

### **Step 1: Create Service Principal (via Azure Portal)**

1. **Go to Azure Portal** → **Azure Active Directory**
2. **App registrations** → **New registration**
3. **Name**: `Detectify-GitHub-Actions`
4. **Account types**: Single tenant
5. **Click "Register"**
6. **Note down**: Application (client) ID, Directory (tenant) ID

### **Step 2: Create Client Secret**

1. **In your app registration** → **Certificates & secrets**
2. **New client secret**
3. **Description**: `GitHub Actions`
4. **Expires**: 24 months
5. **Copy the secret value** (you won't see it again!)

### **Step 3: Assign Permissions**

1. **Go to your Resource Group** (`detectify-rg`)
2. **Access control (IAM)** → **Add role assignment**
3. **Role**: Contributor
4. **Assign access to**: User, group, or service principal
5. **Select**: Your app registration name
6. **Save**

### **Step 4: Add Secrets to GitHub**

Add these three secrets in GitHub → Settings → Secrets:

```
AZUREAPPSERVICE_CLIENTID_[RANDOM] = [Your Application ID]
AZUREAPPSERVICE_TENANTID_[RANDOM] = [Your Directory ID]  
AZUREAPPSERVICE_SUBSCRIPTIONID_[RANDOM] = [Your Subscription ID]
```

**Note**: Azure generates the exact secret names when you set up deployment center.

---

## 🎯 **Which Workflow File to Use**

### **For Beginners (Recommended)**
Use: `simple_azure_deploy.yml`
- Simpler configuration
- Publish profile authentication
- Fewer moving parts

### **For Advanced Users**
Use: `master_Detectify.yml`
- More secure service principal auth
- Separate build and deploy jobs
- Production-ready setup

---

## 🚨 **Troubleshooting Common Issues**

### **Issue 1: Workflow fails with authentication error**
**Solution:**
1. Check if publish profile is correctly copied (entire file content)
2. Verify app name matches exactly
3. Regenerate publish profile if needed

### **Issue 2: "Package deployment using ZIP Deploy failed"**
**Solution:**
1. Check if `startup.sh` file exists
2. Verify Python version (3.11)
3. Check requirements.txt for conflicts

### **Issue 3: Static files not collecting**
**Solution:**
1. Ensure `STATIC_ROOT` is set in settings
2. Check if `static` directory exists
3. Verify environment variables during build

### **Issue 4: Secret not found error**
**Solution:**
1. Check secret names match exactly (case-sensitive)
2. Verify secrets are in the correct repository
3. Check if secret has proper access permissions

---

## 📈 **Workflow Features Included**

### **Automatic Features:**
- ✅ **Python 3.11 setup**
- ✅ **Dependencies installation**
- ✅ **Static files collection**
- ✅ **ZIP packaging for deployment**
- ✅ **Automatic deployment on push to master**
- ✅ **Manual deployment trigger**

### **Optional Features:**
- 🔧 **Test running** (commented out - add your tests)
- 🔧 **Multiple environments** (staging/production)
- 🔧 **Slack notifications** (can be added)
- 🔧 **Database migrations** (runs via startup script)

---

## ✅ **Final Checklist**

Before pushing your workflow:

- [ ] **Choose authentication method** (Publish Profile recommended)
- [ ] **Add required secrets** to GitHub
- [ ] **Update app name** in workflow file
- [ ] **Test local build** works
- [ ] **Verify startup.sh** exists and is executable
- [ ] **Check requirements.txt** is complete
- [ ] **Confirm production.py** settings are correct

---

## 🎉 **Success Indicators**

Your workflow is working when you see:

1. ✅ **Green checkmarks** in GitHub Actions tab
2. ✅ **Successful deployment** logs
3. ✅ **App accessible** at your Azure URL
4. ✅ **No errors** in Azure App Service logs

---

## 💡 **Pro Tips**

1. **Use staging slot**: Deploy to staging first, then swap
2. **Monitor deployments**: Check Azure logs after each deploy
3. **Version your releases**: Use Git tags for releases
4. **Rollback ready**: Keep previous version for quick rollback
5. **Notifications**: Add Slack/email notifications for deployment status

---

**Next Step**: Choose Method A (Publish Profile) for easiest setup, then follow the step-by-step guide above! 🚀
