import os 
from getpass import getpass
from dotenv import load_dotenv
from astroquery.mast import Observations
import argparse


#pylint:disable=no-member

def main(target_name, calib_levle=3, project="JWST", outdir=None):
    load_dotenv()
    api_token = os.getenv("MAST_API_TOKEN")
    if api_token is None:
        print("""
        MAST_API_TOKEN not found.
        The token was not detected in your environment variables or a .env file.
    
        Please enter your token below to continue.

        If you need to generate a token, visit:
        https://auth.mast.stsci.edu/token?suggested_name=Astroquery&suggested_scope=mast:exclusive_access
        """)
        
        api_token = getpass(prompt="Token: ")
    else:
        print("Successfully loaded API token.")

    session = Observations.login(api_token)
    sessioninfo = Observations.session_info()
    #print(sessioninfo)
    obs_list = Observations.query_criteria(target_name=target_name, dataproduct_type="image", calib_level=calib_levle, project=project)
    df = obs_list.to_pandas()
    if outdir is None:
        outdir=""
    else:
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{target_name}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", required=True)
    parser.add_argument("--calib_level", default=3)
    parser.add_argument("--project", default="JWST")
    parser.add_argument("--outdir", default="csv")
    args = parser.parse_args()
    main(args.target_name, args.calib_level, args.project, args.outdir)