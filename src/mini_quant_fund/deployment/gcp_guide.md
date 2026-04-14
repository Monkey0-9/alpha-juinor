# GCP Deployment Guide (Free Tier)

## Option 1: Compute Engine (Always Free)

### Free Tier: e2-micro in us-west1, us-central1, or us-east1

### Step 1: Create VM Instance
```bash
# Via gcloud CLI
gcloud compute instances create mini-quant-fund \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --tags=quant-fund
```

### Step 2: SSH to Instance
```bash
gcloud compute ssh mini-quant-fund --zone=us-central1-a
```

### Step 3: Install Dependencies
```bash
# Same as AWS guide
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone and setup
git clone <your-repo>
cd mini-quant-fund
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Create .env file
nano .env
# Add ALPACA_API_KEY, etc.
```

### Step 5: Schedule with systemd + timer
```bash
# Create service file
sudo nano /etc/systemd/system/mini-quant-fund.service
```

#### mini-quant-fund.service:
```ini
[Unit]
Description=Mini Quant Fund
After=network.target

[Service]
Type=oneshot
User=<your-username>
WorkingDirectory=/home/<user>/mini-quant-fund
ExecStart=/home/<user>/mini-quant-fund/.venv/bin/python main.py
StandardOutput=append:/var/log/fund.log
StandardError=append:/var/log/fund.log

[Install]
WantedBy=multi-user.target
```

#### Create timer:
```bash
sudo nano /etc/systemd/system/mini-quant-fund.timer
```

```ini
[Unit]
Description=Run Mini Quant Fund Daily

[Timer]
# Run at 4 PM EST (21:00 UTC) on weekdays
OnCalendar=Mon-Fri 21:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
# Enable and start timer
sudo systemctl enable mini-quant-fund.timer
sudo systemctl start mini-quant-fund.timer

# Check status
sudo systemctl status mini-quant-fund.timer
```

---

## Option 2: Cloud Run (Serverless)

### Free Tier: 2M requests/month, 360,000 vCPU-seconds

### Step 1: Build Container
```bash
# Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/mini-quant-fund .

# Push to Container Registry
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/mini-quant-fund
```

### Step 2: Deploy to Cloud Run
```bash
gcloud run deploy mini-quant-fund \
    --image gcr.io/YOUR_PROJECT_ID/mini-quant-fund \
    --platform managed \
    --region us-central1 \
    --set-env-vars="ALPACA_API_KEY=...,ALPACA_SECRET_KEY=...,ALERT_EMAIL=..." \
    --no-allow-unauthenticated \
    --memory 512Mi \
    --timeout 300s
```

### Step 3: Schedule with Cloud Scheduler
```bash
gcloud scheduler jobs create http mini-quant-fund-daily \
    --schedule="0 21 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://mini-quant-fund-XXXXX.run.app" \
    --http-method=POST \
    --oidc-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com
```

---

## Monitoring

### Cloud Logging (Free Tier: 50GB/month)
```bash
# View logs
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=YOUR_INSTANCE_ID" --limit 50
```

### Monitoring Dashboard (Free)
```bash
# GCP Console > Monitoring > Dashboards
# Add charts for:
# - CPU utilization
# - Memory usage
# - Log-based metrics
```

---

## Cost Estimate (Free Tier)

| Service | Free Tier | Est. Monthly Cost |
|:--------|:----------|:------------------|
| e2-micro VM | 1 instance | $0 (Always Free) |
| Persistent Disk | 30GB | $0 (Free Tier) |
| Cloud Run | 2M requests | $0 |
| Cloud Scheduler | 3 jobs | $0 (3 free) |
| **Total** | | **$0/month** |

---

## Security
1. **Use Secret Manager** (free for small secrets)
2. **Enable OS Login** for SSH
3. **Configure Firewall** rules
4. **Enable VPC Flow Logs** (optional)
