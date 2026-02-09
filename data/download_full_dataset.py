import kagglehub
import shutil
import os
import glob

def download_and_setup():
    print("⬇️  Downloading dataset from Kaggle...")
    
    # 1. Download to local cache
    # This uses the official KaggleHub API to get the latest version
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
    print(f"✅ Downloaded to cache: {path}")

    # 2. Define source and destination
    # The dataset usually contains a file named 'bitstampUSD_...csv'
    # We use glob to find the CSV regardless of the specific date in the filename
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        print("❌ Error: No CSV found in the downloaded dataset.")
        return

    source_file = csv_files[0] # Take the first CSV found
    destination_dir = os.path.dirname(os.path.abspath(__file__)) # The 'data' folder
    destination_file = os.path.join(destination_dir, "btcusd.csv")

    # 3. Copy and Rename
    print(f"🔄 Moving to: {destination_file}")
    shutil.copy(source_file, destination_file)
    
    print("\n🎉 Success! Data is ready.")
    print(f"📁 You can now run: python test_tps.py")

if __name__ == "__main__":
    download_and_setup()
