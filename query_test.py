from astroquery.mast import Observations
from astroquery.mast import Mast

#pylint:disable=no-member


token = "timed out"

my_session = Observations.login(token)
sessioninfo = Observations.session_info()
print(sessioninfo)


target_name="NGC-628"

obs_list = Observations.query_criteria(target_name=target_name, dataproduct_type="image", calib_level=3, project="JWST")
df = obs_list.to_pandas()
df.to_csv(f"{target_name}.csv", index=False)
