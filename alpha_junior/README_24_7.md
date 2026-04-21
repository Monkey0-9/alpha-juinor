# Alpha Junior - 24/7 Setup Guide

Complete guide to run Alpha Junior 24/7 without interruption.

## 🚀 QUICK START (Choose One)

### **Option 1: Simple 24/7 (Recommended)**
Just double-click and leave it running:
```
Double-click:  run_24_7.bat
```
✅ Auto-restarts if it crashes  
✅ Logs everything  
✅ Keep window minimized  

**Access:** http://localhost:5000

---

### **Option 2: Management Console**
Control panel for all options:
```
Double-click:  manage.bat
```
Choose from menu:
- Start 24/7 mode
- Add to Windows startup
- View logs
- Check status
- Stop server

---

### **Option 3: Auto-Start on Windows Boot**
Run once, then it starts automatically every time you log in:
```
Double-click:  add_to_startup.bat
```
✅ Starts automatically  
✅ No need to do anything after boot  

---

### **Option 4: Windows Service (Advanced)**
Runs even when no user is logged in:

**Step 1:** Run as Administrator:
```
Right-click:  install_service.bat → Run as Administrator
```

**Step 2:** Manage service:
```cmd
net start AlphaJunior    # Start
net stop AlphaJunior     # Stop
sc query AlphaJunior     # Check status
```

---

## 📊 Monitoring

### Check if Running
Open browser:
```
http://localhost:5000/api/health
```
Should show: `{"status": "healthy"}`

### View Logs
```
Logs location: logs/alpha_junior.log
```

### Health Monitor
Run separate monitor that checks every 30 seconds:
```
python monitor.py
```

---

## 🔧 Troubleshooting

### Server not starting?
1. Check Python is installed: `python --version`
2. Check port 5000 is free: `netstat -an | findstr 5000`
3. Check logs: `logs/alpha_junior.log`

### Port already in use?
Change port in `app.py` line with `port=5000` to another number like `5001`

### Logs too big?
Logs auto-rotate at 10MB, keeping 5 backups.

---

## 📁 File Structure

```
alpha_junior/
├── app.py                  # Main application
├── run_24_7.bat           # 24/7 auto-restart runner ⭐
├── manage.bat             # Management console ⭐
├── add_to_startup.bat     # Add to Windows startup
├── install_service.bat    # Install as Windows service
├── service_runner.py      # Service backend
├── monitor.py             # Health monitor
├── requirements.txt       # Dependencies
├── logs/                  # Log files
└── alpha_junior.db       # SQLite database
```

---

## 🌐 Access URLs

| URL | Description |
|-----|-------------|
| http://localhost:5000 | Main website |
| http://localhost:5000/api/health | Health check |
| http://localhost:5000/api/funds | List funds |
| http://localhost:5000/api/register | Register user |

---

## 💡 Pro Tips

1. **For Development:** Use `start.bat` (with debug mode)
2. **For Production:** Use `run_24_7.bat` (auto-restart)
3. **For Auto-Start:** Use `add_to_startup.bat` + `run_24_7.bat`
4. **For Server:** Use `install_service.bat` (no user needed)

---

## 🛑 Stop Everything

**To stop the server:**
- Close the `run_24_7.bat` window, OR
- Run `manage.bat` → Option 6, OR
- Run: `taskkill /f /im python.exe`

**To remove from startup:**
Delete shortcut from:
```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
```

---

## 📞 Support

If something doesn't work:
1. Check logs in `logs/` folder
2. Make sure Python 3.11+ is installed
3. Try restarting your computer
4. Check firewall isn't blocking port 5000

**Built for 24/7 operation!** 🚀
