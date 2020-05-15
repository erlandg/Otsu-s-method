import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Otsu's method for optimum global thresholding

L = 256
im = np.array(Image.open('blaklokke.jpg')).astype('float')/(L-1)
# Try with both grey and blue color scale
# Grey:
# im_ = np.array(Image.open('blaklokke.jpg').convert('LA'))[:,:,0].astype('float')/(L-1)
# Blue:
im_ = np.array(Image.open('blaklokke.jpg'))[:,:,2].astype('float')/(L-1)
P = im.shape[0];   Q = im.shape[1]

int, count = np.unique((im_*(L-1)).astype('int'), return_counts = True)
hist = np.zeros(L)
hist[int-1] = count/P/Q

m = lambda k: np.sum([i*hist[i] for i in range(k)])
gl_m = m(L)
gl_v = np.sum([(i-gl_m)**2*hist[i] for i in range(L)])

P = np.array([[np.sum(hist[:k]), np.sum(hist[k:])] for k in range(L)])
m = np.nan_to_num(np.array([[m(k), m(L)-m(k)] for k in range(L)]))

b_v = P[:,0] * (m[:,0] - gl_m)**2 + P[:,1] * (m[:,1] - gl_m)**2
k_ = np.argmax(b_v)
n_ = b_v[k_]/gl_v

plt.bar(np.arange(L), hist)
plt.axvline(x=k_-0.5, color='red')
plt.show()
plt.clf()

im1 = im.copy()
im1[im_ < k_/(L-1)] = 0

plt.subplot(121)
plt.imshow(im)

plt.subplot(122)
plt.imshow(im1)
plt.show()
