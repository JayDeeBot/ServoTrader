import os
import json

def check_crypto_csv_coverage(json_path, csv_folder_path):
    # Load crypto codes from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the list from the "crypto_codes" key
    crypto_codes = data.get("crypto_codes", [])
    print(f"✅ Found {len(crypto_codes)} crypto codes in JSON config\n")

    # Get all CSV filenames (strip .csv)
    csv_files = [f[:-4] for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

    # Track presence
    present = []
    missing = []

    for code in crypto_codes:
        if code in csv_files:
            present.append(code)
        else:
            missing.append(code)

    # Report results
    print(f"✅ {len(present)} codes have corresponding CSVs:\n{present}\n")
    print(f"❌ {len(missing)} codes are missing CSVs:\n{missing}\n")

# Usage
if __name__ == "__main__":
    json_path = "/home/jarred/git/ServoTrader/servo_trader/config/crypto_codes_ancient_2.json"
    csv_folder = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs_ancient_2"
    check_crypto_csv_coverage(json_path, csv_folder)
