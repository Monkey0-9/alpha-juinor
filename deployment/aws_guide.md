# AWS Deployment Guide (Free Tier)

## Option 1: EC2 (Recommended for Always-On)

### Free Tier: 750 hours/month t2.micro (1 vCPU, 1GB RAM)

### Step 1: Launch EC2 Instance
```bash
# 1. Go to AWS Console > EC2 > Launch Instance
# 2. Select: Ubuntu Server 22.04 LTS (Free Tier)
# 3. Instance Type: t2.micro
# 4. Key pair: Create new (download .pem file)
# 5. Security Group: Allow SSH (port 22) from your IP
# 6. Storage: 30GB gp3 (Free Tier: 30GB)
```

### Step 2: Connect to Instance
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<your-ec2-ip>
```

### Step 3: Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

### Step 4: Deploy Application
```bash
# Clone repository
git clone <your-repo-url>
cd mini-quant-fund

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
nano .env
```

#### .env file:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALERT_EMAIL=your@gmail.com
ALERT_PASSWORD=your_app_password
ALERT_TO=alerts@yourmail.com
```

### Step 5: Schedule with Cron
```bash
# Edit crontab
crontab -e

# Run daily at 4 PM EST (after market close)
0 21 * * 1-5 cd /home/ubuntu/mini-quant-fund && .venv/bin/python main.py >> /var/log/fund.log 2>&1
```

### Step 6: Monitor
```bash
# View logs
tail -f /var/log/fund.log

# Check system resources
htop
```

---

## Option 2: Lambda + EventBridge (Serverless)

### Free Tier: 1M requests/month, 400,000 GB-seconds compute

### Step 1: Create Lambda Function
```bash
# Package code
zip -r function.zip . -x "*.git*" -x "tests/*" -x ".venv/*"

# Upload to Lambda via AWS Console
# Runtime: Python 3.11
# Memory: 512MB (sufficient for our workload)
# Timeout: 5 minutes
```

### Step 2: Add Environment Variables
In Lambda Console > Configuration > Environment variables:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALERT_EMAIL=...
ALERT_PASSWORD=...
ALERT_TO=...
```

### Step 3: Create EventBridge Rule
```bash
# Schedule expression (daily at 4 PM EST):
cron(0 21 ? * MON-FRI *)

# Target: Your Lambda function
```

### Step 4: Monitor
```bash
# CloudWatch Logs (Free Tier: 5GB ingestion/month)
aws logs tail /aws/lambda/mini-quant-fund --follow
```

---

## Cost Estimate (Free Tier)

| Service | Free Tier | Est. Monthly Cost |
|:--------|:----------|:------------------|
| EC2 t2.micro | 750 hrs | $0 (within limit) |
| EBS 30GB | 30GB | $0 |
| Lambda | 1M requests | $0 |
| CloudWatch Logs | 5GB | $0 |
| **Total** | | **$0/month** |

---

## Security Best Practices
1. **Never commit API keys** - Use .env files (add to .gitignore)
2. **Enable MFA** on AWS account
3. **Restrict Security Groups** to your IP only
4. **Rotate credentials** quarterly
5. **Enable CloudTrail** for audit logging (free)
