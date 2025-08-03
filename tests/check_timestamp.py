from datetime import datetime, timezone

def check_timestamp(ts):
    try:
        ts = int(ts)
        readable = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        print(f"✅ UNIX Timestamp: {ts}")
        print(f"🕒 Readable UTC Time: {readable}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    user_input = input("Enter a UNIX timestamp (e.g., from CSV): ")
    check_timestamp(user_input)
