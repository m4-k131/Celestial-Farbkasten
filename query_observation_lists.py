import os 
from dotenv import load_dotenv
from astroquery.mast import Observations
import argparse


#pylint:disable=no-member

def main(target_name, calib_levle=3, project="JWST", outdir=None):
    load_dotenv()
    api_token = os.getenv("MAST_API_TOKEN")
    if api_token is None:
        print("Error: MAST_API_TOKEN not found in .env file")
    else:
        print("Successfully loaded API token.")
    session = Observations.login(api_token)
    sessioninfo = Observations.session_info()
    print(sessioninfo)
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
    parser.add_argument("--outdir", default="outs_csv")
    args = parser.parse_args()
    main(args.target_name, args.calib_level, args.project, args.outdir)