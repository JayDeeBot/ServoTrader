# 🚀 ServoTrader Deployment Guide (DigitalOcean)

## 1. Create the Droplet
- Provider: **DigitalOcean**
- Recommended size: `2 GB RAM` minimum (upgrade if OOM issues).
- Region: nearest to you or Binance servers (e.g. Singapore).

SSH into the server:
```bash
ssh root@<YOUR_DROPLET_IP>
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
MODEL_PATH=/app/models/ppo_servo_trader_squirtle.zip
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
scp ./config/crypto_codes.json root@<DROPLET_IP>:/srv/servotrader/config/
scp ./config/params.yaml root@<DROPLET_IP>:/srv/servotrader/config/
scp -r ./models root@<DROPLET_IP>:/srv/servotrader/
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
http://<YOUR_DROPLET_IP>
```

---

✅ Done — ServoTrader is now live!  
If you reboot the droplet, Docker will restart it automatically (`--restart unless-stopped`).  