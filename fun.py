import pandas as pd

def get_rig_count():
    # Baker Hughes direct link to their latest Excel file
    url = "https://rigcount.bakerhughes.com/static-files/7203f16d-3121-4835-8664-90a4282e70e9" 
    # Note: This URL changes occasionally; institutional desks often scrape the landing page first.
    
    try:
        # Read the 'Summary' or 'Master' sheet
        df = pd.read_excel(url, sheet_name='Master Data', skiprows=6)
        # Filter for 'United States' and 'Oil'
        oil_rigs = df[df['Country'] == 'United States'][['Publish Date', 'Oil']]
        print(oil_rigs)
        return oil_rigs
    except Exception as e:
        print(f"Error fetching Rigs: {e}")
        return None
    

get_rig_count()