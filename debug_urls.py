import requests
import json

# Let's use an ID from one of your failing files: 'jw02107-c1019_t018_miri_f2100w_i2d.fits'
# The core observation ID is 'jw02107c1019'
OBSERVATION_ID = "jw02107c1019"

# This is the MAST API endpoint for querying observations
mast_api_url = "https://mast.stsci.edu/api/v0.1/portal_search.json"

# Construct the query payload
# We are asking for all products associated with a specific observation ID
params = {
    "search_text": json.dumps({
        "columns": "*",
        "filters": [
            {
                "paramName": "obs_id",
                "values": [OBSERVATION_ID]
            }
        ]
    })
}

print(f"Querying MAST for observation ID: {OBSERVATION_ID}...")
response = requests.get(mast_api_url, params=params)
response.raise_for_status() # Will raise an error if the request failed

search_results = response.json()
products = search_results.get('data', [])

if not products:
    print("No products found for this observation ID.")
else:
    print(f"Found {len(products)} associated data products.")
    # Print the details for the first FITS science product found
    for product in products:
        if product.get("productType") == "SCIENCE" and product.get("extension") == "fits":
            print("\n--- Found a Valid Science Product ---")
            print(f"Description: {product.get('description')}")
            
            # THIS is the current, valid URI. Compare it to your CSV.
            print(f"VALID dataURI: {product.get('dataURI')}")
            
            # THIS is often a direct download link to the cloud.
            print(f"DIRECT dataURL: {product.get('dataURL')}")
            print("-------------------------------------")
            break # Stop after finding the first one