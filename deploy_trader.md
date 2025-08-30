# 🚀 ServoTrader Deployment Guide (DigitalOcean)

## Pre-requisites 

1. Build the container with Docker:

```bash
docker compose build
```

2. Run the container to test:

```bash
docker compose up -d
```

3. Check the logs:

```bash
docker compose logs -f
```

4. Stop the container:

```bash
docker compose down
```

5. Push the container to docker:

```bash
docker login
docker build -t jaydeebot/servotrader:latest .
docker push jaydeebot/servotrader:latest
```

## 1. Create the Droplet
- Provider: **DigitalOcean**
- Recommended size: `2 GB RAM` minimum (upgrade if OOM issues).
- Region: nearest to you or Binance servers (e.g. Singapore).

SSH into the server:
```bash
ssh root@<143.198.195.92>
```

---

## 2. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
```

Verify install:
```bash
docker version
```

---

## 3. Create App Folders
```bash
mkdir -p /srv/servotrader/{config,data,logs,models}
```

---

## 4. Create `.env` File  
File: `/srv/servotrader/.env`

```env
MODEL_PATH=/app/models/ppo_bulbasaur.zip
CRYPTO_CODES_PATH=/app/config/crypto_codes.json
PARAMS_PATH=/app/config/params.yaml
DATA_DIR=/app/data
LOG_DIR=/app/logs

# 🔑 Your Binance keys (no quotes!)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here

# (optional) enforce mainnet
BINANCE_TESTNET=false
```

---

## 5. Upload Config + Model
From your local machine:
```bash
scp ./servo_trader/config/crypto_codes.json root@143.198.195.92:/srv/servotrader/config/
scp ./servo_trader/config/params.yaml root@143.198.195.92:/srv/servotrader/config/
scp -r ./models root@143.198.195.92:/srv/servotrader/
```

---

## 6. Fix Permissions for Logs & Data
```bash
chown -R 1000:1000 /srv/servotrader/logs
chown -R 1000:1000 /srv/servotrader/data
```

---

## 7. Add Swap Space (Prevents OOM Kills)
```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

Check:
```bash
free -h
```

---

## 8. Run the Container
```bash
docker login

docker pull jaydeebot/servotrader-live:latest

docker run -d --name servotrader   --restart unless-stopped   --env-file /srv/servotrader/.env   -v /srv/servotrader/config:/app/config   -v /srv/servotrader/data:/app/data   -v /srv/servotrader/logs:/app/logs   -v /srv/servotrader/models:/app/models   -v /srv/servotrader/logs:/home/jarred/git/ServoTrader/logs   -v /srv/servotrader/data:/home/jarred/git/ServoTrader/data   -p 80:8080   jaydeebot/servotrader-live:latest
```

---

## 9. Verify Deployment
### Check logs:
```bash
docker logs -f servotrader
```

### Check API heartbeat:
```bash
apt update && apt install -y jq   # only first time
curl -s http://127.0.0.1/api/live | jq .Step
```

Step should increment every few seconds.

---

## 10. Access Web UI
Open in browser:
```
http://<143.198.195.92>
```

---

✅ Done — ServoTrader is now live!  
If you reboot the droplet, Docker will restart it automatically (`--restart unless-stopped`).  