import codecs
import numpy as np

files = ["ru_embd", "en_embd"]
nfiles = []
dim = 1024

for f in files:
	fraw = f + ".raw"
	fbin = f + ".bin"

	X = np.fromfile(fraw, dtype=np.float32, count=-1)                                                                          
	X.resize(X.shape[0] // dim, dim)

	X.tofile(fbin)