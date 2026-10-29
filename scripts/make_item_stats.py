import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from paiec import features as F
x = pd.read_parquet('data/proc/code_items.parquet').sort_values('i')
S = np.stack([F.item_stats(c, f, '') for c, f in zip(x.content, x.item_features)])
np.save('data/proc/item_stats.npy', S.astype(np.float32)); print(S.shape)
