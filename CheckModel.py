from numpy import load
from os.path import join as join
data = load('imagenet21k_R26+ViT-B_32.npz')
lst = data.files
for item in lst:
    print(item)
    # print(data[item])
