# Alpha Junior - Quick Start Guide

## 🚀 Run the Project Now

### Prerequisites
- Docker Desktop installed and running
- Git (optional, for cloning)

### Step 1: Start the Platform

**Windows:**
```bash
cd c:\mini-quant-fund\alpha_junior
start.bat
```

**Linux/Mac:**
```bash
cd alpha_junior
chmod +x start.sh
./start.sh
```

**Or manually with Docker:**
```bash
docker-compose up -d
```

### Step 2: Wait for Services (30 seconds)

Check status:
```bash
docker-compose ps
```

All containers should show "Up":
- aj_postgres
- aj_redis  
- aj_backend
- aj_frontend
- aj_celery_worker
- aj_celery_beat

### Step 3: Access the Platform

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/api/v1/docs |
| **Flower** | http://localhost:5555 |

---

## 🧪 Test the API

### 1. Register a User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123!","full_name":"Test User"}'
```

### 2. Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123!"}'
```

### 3. Test Market Data (No Auth Required)
```bash
curl http://localhost:8000/api/v1/market/quote/AAPL
```

---

## 🛠️ Common Issues & Fixes

### Issue: "Cannot connect to database"
**Fix:** Wait 10 seconds for PostgreSQL to start, then restart backend:
```bash
docker-compose restart backend
```

### Issue: "Import errors"
**Fix:** Check that all models are properly imported. Run the test:
```bash
cd backend
python test_imports.py
```

### Issue: "Port already in use"
**Fix:** Kill existing containers or change ports in docker-compose.yml:
```bash
docker-compose down
docker-compose up -d
```

### Issue: "Permission denied" (Linux/Mac)
**Fix:** 
```bash
chmod +x start.sh
sudo chown -R $USER:$USER .
```

---

## 📊 View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

## 🛑 Stop Everything

```bash
docker-compose down
```

To remove all data (including database):
```bash
docker-compose down -v
```

---

## 🔧 Development Mode

### Backend Only
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Only
```bash
cd frontend
npm install
npm run dev
```

---

## ✅ Verification Checklist

- [ ] All containers running (`docker-compose ps`)
- [ ] Frontend loads at http://localhost:3000
- [ ] API docs load at http://localhost:8000/api/v1/docs
- [ ] Can register a new user
- [ ] Can login and get token
- [ ] Market data endpoints work

---

## 🆘 Still Having Issues?

Check the detailed logs:
```bash
docker-compose logs backend > backend_logs.txt
docker-compose logs frontend > frontend_logs.txt
```

Then run the import test to find specific errors:
```bash
cd backend
python -c "from app.main import app; print('OK')"
```

**Contact:** Support details in README.md
