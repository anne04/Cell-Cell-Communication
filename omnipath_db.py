# https://workflows.omnipathdb.org/intercell-networks-py.html
import omnipath as op
import pandas as pd


intercell = op.interactions.import_intercell_network()

intercell_filtered = intercell[
    (intercell['category_intercell_source'] == 'ligand'  | intercell['category_intercell_source'] == 'secreted' ) & # set transmitters to be ligands
    (intercell['category_intercell_target'] == 'receptor') # set receivers to be receptors
]
intercell_filtered
intercell_filtered.to_csv('omnipath_lr_db.csv', index=False)
