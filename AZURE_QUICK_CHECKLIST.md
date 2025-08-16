# ✅ Azure Portal Deployment Checklist - Detectify

## 📋 Quick Reference Guide

### Phase 1: Setup Azure Resources
- [ ] **Create Resource Group** `detectify-rg`
- [ ] **Create PostgreSQL Database**
  - [ ] Server: `detectify-db-server`
  - [ ] Database: `detectify`
  - [ ] User: `detectify_admin`
  - [ ] Configure firewall rules
- [ ] **Create Redis Cache** (optional)
  - [ ] Name: `detectify-redis`
  - [ ] Tier: Basic C0
- [ ] **Create App Service**
  - [ ] Name: `detectify-app-[unique]`
  - [ ] Runtime: Python 3.11
  - [ ] Plan: F1 (Free) or B1 (Basic)

### Phase 2: Deploy Code
- [ ] **Connect GitHub Repository**
  - [ ] Source: GitHub
  - [ ] Repository: `Detectify_D`
  - [ ] Branch: `master`
- [ ] **Set Startup Command**
  ```bash
  bash startup.sh
  ```

### Phase 3: Configure Environment Variables
#### Django Settings:
- [ ] `DJANGO_SETTINGS_MODULE` = `detectify_project.settings.production`
- [ ] `SECRET_KEY` = `[Generate new secure key]`

#### Database Settings:
- [ ] `AZURE_POSTGRESQL_NAME` = `detectify`
- [ ] `AZURE_POSTGRESQL_USER` = `detectify_admin@detectify-db-server`
- [ ] `AZURE_POSTGRESQL_PASSWORD` = `[Your DB password]`
- [ ] `AZURE_POSTGRESQL_HOST` = `detectify-db-server.postgres.database.azure.com`
- [ ] `AZURE_POSTGRESQL_PORT` = `5432`

#### Redis Settings (if using Redis):
- [ ] `REDIS_URL` = `[Redis connection string from portal]`

### Phase 4: Initial Setup
- [ ] **Trigger Deployment**
  - [ ] Go to Deployment Center → Sync
  - [ ] Monitor deployment logs
- [ ] **Run Database Setup**
  - [ ] Open SSH terminal
  - [ ] Run migrations: `python manage.py migrate`
  - [ ] Collect static files: `python manage.py collectstatic --noinput`
  - [ ] Create superuser: `python manage.py create_azure_superuser`

### Phase 5: Testing & Verification
- [ ] **Test Application**
  - [ ] Visit your app URL
  - [ ] Test homepage loading
  - [ ] Test admin login at `/admin/`
- [ ] **Test Core Features**
  - [ ] User registration/login
  - [ ] Case creation
  - [ ] File uploads (if working)
  - [ ] Database connectivity

### Phase 6: Production Readiness
- [ ] **Security**
  - [ ] Change default admin password
  - [ ] Verify HTTPS is working
  - [ ] Test environment variables
- [ ] **Monitoring**
  - [ ] Enable Application Insights
  - [ ] Configure log monitoring
  - [ ] Set up alerts
- [ ] **Performance**
  - [ ] Test loading times
  - [ ] Monitor resource usage
  - [ ] Consider scaling if needed

## 🚨 Common Issues & Quick Fixes

### Issue: "Application Error" Page
**Quick Fix:**
1. Go to App Service → Log stream
2. Check for Python/Django errors
3. Verify environment variables are set correctly

### Issue: Database Connection Failed
**Quick Fix:**
1. Check PostgreSQL firewall settings
2. Verify connection string format
3. Test connection from App Service SSH

### Issue: Static Files Not Loading
**Quick Fix:**
1. SSH into app: `cd /home/site/wwwroot`
2. Run: `python manage.py collectstatic --noinput`
3. Check if `staticfiles` directory was created

### Issue: Deployment Stuck/Failed
**Quick Fix:**
1. Go to Deployment Center
2. Check deployment logs for errors
3. Try manual sync or redeploy

## 📞 Getting Help

### Azure Portal Resources:
- **Resource Health**: Check service status
- **Support + Troubleshooting**: Get help with issues
- **Advisor**: Performance recommendations
- **Cost Management**: Monitor spending

### Documentation Links:
- [App Service Python Apps](https://docs.microsoft.com/azure/app-service/configure-language-python)
- [PostgreSQL on Azure](https://docs.microsoft.com/azure/postgresql/)
- [Redis Cache](https://docs.microsoft.com/azure/azure-cache-for-redis/)

## 🎯 Success Criteria

Your deployment is successful when:
- ✅ Application loads without errors
- ✅ Admin panel accessible
- ✅ Database operations work
- ✅ Static files display correctly
- ✅ HTTPS certificate active
- ✅ Logs show no critical errors

## 💡 Pro Tips

1. **Use Deployment Slots**: Create staging slot for testing
2. **Monitor Costs**: Set up budget alerts
3. **Backup Strategy**: Configure automated backups
4. **Custom Domain**: Add your domain for production
5. **CDN**: Use Azure CDN for better performance

---

**Estimated Total Time**: 2-3 hours for complete setup
**Recommended Order**: Follow phases 1-6 sequentially for best results

*Good luck with your Azure deployment! 🚀*
