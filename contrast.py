import numpy as np
import time

a = np.random.rand(128, 32)
b = np.random.rand(32, 128)
start_time = time.time()
c = np.dot(b, a)
print(time.time()-start_time)
print(c.shape)

