import sys, pickle; sys.path.insert(0, '.')
import pandas as pd
from paiec.tokens import numeric_features
x = pd.read_parquet('data/proc/code_items.parquet').sort_values('i')
pickle.dump([numeric_features(f) for f in x.item_features], open('data/proc/item_numfeat.pkl', 'wb'))
print('ok')
