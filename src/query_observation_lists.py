import os 
from getpass import getpass
from dotenv import load_dotenv
from astroquery.mast import Observations
import argparse

from paths import CSV_DIR


#pylint:disable=no-member

def main(target_name, exact_name=False, calib_levle=3, project="JWST", outdir=None):
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
    if not exact_name:
        target_name=f"*{target_name}*"
    obs_list = Observations.query_criteria(target_name=target_name, dataproduct_type="image", calib_level=calib_levle, project=project)
    target_name=target_name.strip("*")
    df = obs_list.to_pandas()
    if outdir is None:
        outdir=CSV_DIR
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{target_name if not exact_name else target_name+'_actual'}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", required=True)
    parser.add_argument("--exact_name", action="store_true")
    parser.add_argument("--calib_level", default=3)
    parser.add_argument("--project", default="JWST")
    parser.add_argument("--outdir", default=CSV_DIR)
    args = parser.parse_args()
    main(args.target_name, args.exact_name, args.calib_level, args.project, args.outdir)