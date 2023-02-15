from omnipath._core.requests.interactions._utils import import_intercell_network
import pandas as pd

a = import_intercell_network()
df = pd.DataFrame(a)
df.to_csv('/cluster/home/t116508uhn/64630/omnipath_records_2023Feb.csv', index=False)
