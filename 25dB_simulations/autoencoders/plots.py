import matplotlib.pyplot as plt
import numpy as np

from res.gyro_output_5 import y
from res.gyro_real_output_5 import yr

npy = np.array(y)
npyr = np.array(yr)

plt.subplot(2, 1, 1)
j = 5000
plt.plot(npy[1:j], label = 'y')
plt.plot(npy[1:j] - npyr[1:j], label = 'y-\hat y')
plt.legend()

plt.subplot(2, 1, 2)
i = 5000
plt.plot(npy[1:i], label = 'y')
plt.plot(npyr[1:i], label = '\hat y')
plt.legend()

plt.show()

from res.wh_output import y
from res.wh_real_output import yr

npy = np.array(y)
npyr = np.array(yr)

plt.subplot(2, 1, 1)
j = 5000
plt.plot(npy[1:j], label = 'y')
plt.plot(npy[1:j] - npyr[1:j], label = 'y-\hat y')
plt.legend()

plt.subplot(2, 1, 2)
i = 5000
plt.plot(npy[1:i], label = 'y')
plt.plot(npyr[1:i], label = '\hat y')
plt.legend()

plt.show()

from res.gyro_output_10 import y
from res.gyro_real_output_10 import yr

npy = np.array(y)
npyr = np.array(yr)

plt.subplot(2, 1, 1)
j = 5000
plt.plot(npy[1:j], label = 'y')
plt.plot(npy[1:j] - npyr[1:j], label = 'y-\hat y')
plt.legend()

plt.subplot(2, 1, 2)
i = 5000
plt.plot(npy[1:i], label = 'y')
plt.plot(npyr[1:i], label = '\hat y')
plt.legend()

plt.show()
