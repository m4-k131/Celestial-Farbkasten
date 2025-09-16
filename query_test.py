from astroquery.mast import Observations
from astroquery.mast import Mast
from download_utils import download_with_tqdm, download_file_verbose

#pylint:disable=no-member


token = "9bc4249132e7467e851c6d24f86bc96b"

my_session = Observations.login(token)
sessioninfo = Observations.session_info()
print(sessioninfo)

"""
obs_list = Observations.query_criteria(proposal_id=2733)
# We can chooose the columns we want to display in our table
disp_col = ['dataproduct_type', 'calib_level', 'obs_id',
            'target_name', 'filters', 'proposal_pi', 'obs_collection']
obs_list[disp_col]
mask = (obs_list['obs_id'] == 'jw02733-o002_t001_miri_f1130w')
data_products = Observations.get_product_list(obs_list[mask])
print(len(data_products))
filtered_prod = Observations.filter_products(data_products, calib_level=[2], productType="SCIENCE")
# Again, we choose columns of interest for convenience
disp_col = ['obsID', 'dataproduct_type', 'productFilename', 'size', 'calib_level']
filtered_prod[disp_col][:10]
# Step 1: Broaden the search by removing the product type filter.
manifest = Observations.download_products(filtered_prod[:5])
print(manifest['Status'])
"""

#obs_list = Observations.query_criteria(target_name="M74")

target_name="OMC2-SE"

obs_list = Observations.query_criteria(target_name=target_name, dataproduct_type="image", calib_level=3, project="JWST")
df = obs_list.to_pandas()
df.to_csv(f"{target_name}.csv", index=False)
