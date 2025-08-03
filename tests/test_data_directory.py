import os

def count_csv_files_and_lines(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"✅ Found {len(csv_files)} CSV files in '{folder_path}'\n")

    # Count lines in each file
    for file_name in sorted(csv_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r') as file:
                line_count = sum(1 for _ in file) - 1  # Subtract header line
            print(f"📄 {file_name}: {line_count} data lines")
        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

# Usage
if __name__ == "__main__":
    folder = "/home/jarred/git/ServoTrader/data/individual_crypto_csvs"
    count_csv_files_and_lines(folder)
