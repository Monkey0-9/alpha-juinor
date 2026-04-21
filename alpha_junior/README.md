# Alpha Junior - Institutional Fund Platform

A world-class, zero-compromise fund management platform competing at the top 1% level of global fund platforms. Built with Python 3.12+, FastAPI, Next.js 14, PostgreSQL, and Redis.

![Alpha Junior](https://img.shields.io/badge/Alpha%20Junior-Institutional%20Grade-blue)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 What Makes This Top 1%

✅ **Institutional Security**: RS256 JWT, 2FA TOTP, bcrypt cost 12, rate limiting, audit logging  
✅ **Production-Grade Architecture**: Async FastAPI, SQLAlchemy 2.0, Redis caching, PostgreSQL  
✅ **Real Market Data**: Alpha Vantage, CoinGecko, FRED, NewsAPI integration  
✅ **Professional Frontend**: Next.js 14, Tailwind CSS, institutional design system  
✅ **DevOps Ready**: Docker Compose, Nginx, Prometheus, Grafana, Celery workers  
✅ **Compliance**: KYC/AML workflows, audit trails, accredited investor verification  

---

## 📁 Project Structure

```
alpha_junior/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/   # REST API routes
│   │   ├── core/               # Config, security, deps
│   │   ├── db/                 # Database session
│   │   ├── models/             # SQLAlchemy models (13 tables)
│   │   ├── services/           # External API integrations
│   │   ├── tasks/              # Celery background jobs
│   │   └── main.py             # FastAPI app
│   ├── Dockerfile
│   ├── requirements.txt
│   └── alembic/                # Database migrations
├── frontend/                   # Next.js 14 Frontend
│   ├── app/                    # App router
│   ├── components/             # React components
│   ├── lib/                    # API client, utilities
│   ├── Dockerfile
│   └── package.json
├── nginx/                      # Reverse proxy config
├── monitoring/                 # Prometheus & Grafana
├── docker-compose.yml          # Full stack orchestration
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 20+ (for local frontend dev)
- Python 3.12+ (for local backend dev)
- Free API keys (Alpha Vantage, NewsAPI, FRED)

### 1. Clone & Configure

```bash
git clone <repository-url>
cd alpha_junior

# Copy environment template
cp .env.example .env

# Edit .env with your API keys and secrets
nano .env
```

### 2. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

### 3. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Next.js application |
| Backend API | http://localhost:8000/api/v1 | FastAPI docs at `/docs` |
| Flower | http://localhost:5555 | Celery task monitor |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3001 | Dashboards (admin/admin) |

---

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: Async-first, OpenAPI auto-docs, Pydantic validation
- **SQLAlchemy 2.0**: Async ORM with proper type annotations
- **PostgreSQL 16**: Primary database with JSONB for flexibility
- **Redis 7**: Caching, session store, Celery broker
- **Celery**: Background task processing (NAV calc, email, reports)
- **JWT**: RS256 asymmetric signing with refresh token rotation
- **2FA**: TOTP-based two-factor authentication

### Frontend Stack
- **Next.js 14**: App router, SSR for SEO, API routes
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first styling with custom design system
- **TanStack Query**: Server state management, caching
- **Zustand**: Client state management
- **Recharts**: Data visualization
- **Radix UI**: Accessible component primitives

### External APIs
- **Alpha Vantage**: Stock quotes, historical prices, fundamentals
- **CoinGecko**: Cryptocurrency prices and market data
- **ExchangeRate-API**: Currency conversion (free, no key)
- **FRED**: Economic indicators (Fed rate, inflation, unemployment)
- **NewsAPI**: Financial news for sentiment analysis

---

## 📊 Database Schema (13 Tables)

### Core Tables
| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | Authentication | JWT, 2FA, role-based access, soft delete |
| `user_profiles` | Investor details | Accreditation, net worth, bank details |
| `funds` | Fund management | NAV tracking, strategy, status, slug |
| `fund_performance` | Historical data | Daily NAV, Sharpe, alpha, beta, benchmark |
| `investments` | Investment lifecycle | Units, entry NAV, P&L, redemption |
| `kyc_submissions` | Identity verification | S3 documents, AI scores, review workflow |
| `audit_logs` | Compliance | Immutable trail, before/after payloads |

---

## 🔐 Security Features

- ✅ **Passwords**: bcrypt with cost factor 12
- ✅ **Tokens**: RS256 JWT, 15-min access, 7-day refresh with rotation
- ✅ **2FA**: TOTP with QR code generation
- ✅ **Rate Limiting**: 5 failed logins → 15-min lockout
- ✅ **Input Validation**: Pydantic v2 on every request
- ✅ **SQL Injection**: SQLAlchemy parameterized queries only
- ✅ **CORS**: Whitelist-based origin validation
- ✅ **Secrets**: Environment variables, never in code
- ✅ **Audit**: Every state change logged with IP and user agent

---

## 📈 API Endpoints

### Authentication
```
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/verify-email
POST /api/v1/auth/refresh
POST /api/v1/auth/logout
POST /api/v1/auth/enable-2fa
```

### Funds
```
GET  /api/v1/funds              # Discovery with filters
GET  /api/v1/funds/{slug}       # Fund detail
POST /api/v1/funds              # Create fund (manager)
GET  /api/v1/funds/{id}/performance
```

### Investments
```
GET  /api/v1/investments/portfolio
POST /api/v1/investments/subscribe
POST /api/v1/investments/{id}/approve
```

### Market Data
```
GET  /api/v1/market/quote/{symbol}
GET  /api/v1/market/benchmarks
GET  /api/v1/market/crypto/{coin}
POST /api/v1/market/currency/convert
GET  /api/v1/market/economic-indicators
```

---

## 🔄 Background Jobs (Celery)

| Task | Schedule | Description |
|------|----------|-------------|
| `update_all_fund_navs` | 4:30 PM ET daily | Calculate NAV from holdings |
| `fetch_benchmark_data` | Every 5 min | S&P 500, NASDAQ, Dow, VIX |
| `fetch_crypto_data` | Every 5 min | Top 100 crypto prices |
| `fetch_economic_indicators` | Every hour | Fed rate, inflation, unemployment |
| `fetch_financial_news` | Every 30 min | Market sentiment analysis |

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pip install -r requirements.txt
pytest --cov=app tests/

# Frontend tests
cd frontend
npm install
npm run test
```

---

## 🚀 Production Deployment

### 1. SSL Certificates
```bash
# Generate self-signed for testing
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# For production, use Let's Encrypt
```

### 2. Environment Variables
```bash
# Production settings
NODE_ENV=production
LOG_LEVEL=WARNING
DEBUG=false
```

### 3. Deploy with Docker
```bash
# Production profile (includes nginx, prometheus, grafana)
docker-compose --profile production --profile monitoring up -d
```

### 4. Database Migrations
```bash
# Run Alembic migrations
docker-compose exec backend alembic upgrade head
```

---

## 📚 Documentation

- [API Documentation](http://localhost:8000/api/v1/docs) - Swagger UI
- [External APIs](backend/EXTERNAL_APIS.md) - API keys and usage
- [Database Schema](backend/alembic/) - Migration files
- [Monitoring](monitoring/) - Prometheus & Grafana setup

---

## 🎨 Design System

### Colors
- **Base**: `#0A0F1E` (Deep navy)
- **Primary**: `#4F46E5` (Electric indigo)
- **Accent**: `#F59E0B` (Soft gold)
- **Success**: `#10B981`
- **Danger**: `#EF4444`

### Typography
- **Body**: Inter
- **Headings**: Sora
- **Monospace**: JetBrains Mono

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**This is a demonstration platform. Not financial advice.**

- Past performance is not indicative of future results
- This platform does not provide investment advice
- Consult with qualified financial advisors before investing
- Investment involves risk, including loss of principal

---

## 🙏 Acknowledgments

- [Public APIs](https://github.com/public-apis/public-apis) for market data sources
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [SQLAlchemy](https://www.sqlalchemy.org/) for the ORM

---

**Built with ❤️ by the Alpha Junior Engineering Team**

For support, contact: support@alphajunior.com
