"""Standardize each dimension over all items, then L2-normalize rows (float16 output)."""
import sys
import numpy as np
for path in sys.argv[1:]:
    X = np.load(path).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    out = path.replace('.npy', '_z.npy')
    np.save(out, X.astype(np.float16)); print(out, X.shape)
