import os
import requests
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
API_KEY = os.getenv('FRED_API_KEY')

if not API_KEY:
    print("❌ ERROR: FRED_API_KEY not found in .env file!")
    exit()

# 2. Official Baker Hughes Series IDs on FRED
SERIES_MAP = {
    'Total': 'ROTARYRIGUS',
    'Oil': 'ROTARYRIGOILUS',
    'Gas': 'ROTARYRIGGASUS'
}

def fetch_fred_data(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': API_KEY,
        'file_type': 'json',
        'sort_order': 'desc',
        'limit': 2 
    }
    
    try:
        response = requests.get(url, params=params)
        # If the API key is wrong, this will print the reason
        if response.status_code != 200:
            print(f"⚠️ API Error for {series_id}: {response.json().get('error_message')}")
            return None
            
        return response.json().get('observations', [])
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# 3. Process and Print Summary
print(f"\n{'CATEGORY':<10} | {'LATEST':<8} | {'PREVIOUS':<8} | {'CHANGE'}")
print("-" * 48)

latest_date = "N/A"

for label, s_id in SERIES_MAP.items():
    data = fetch_fred_data(s_id)
    
    if data and len(data) >= 2:
        current_val = int(data['value'])
        prev_val = int(data['value'])
        diff = current_val - prev_val
        latest_date = data['date']
        
        change_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{label:<10} | {current_val:<8} | {prev_val:<8} | {change_str}")
    else:
        print(f"{label:<10} | Fetch Failed")

print("-" * 48)
print(f"Data Source: Baker Hughes via FRED | Updated: {latest_date}\n")