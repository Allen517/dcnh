# -*- coding=UTF-8 -*-\n
import numpy as np
import random

import json
import re

import matplotlib.pyplot as plt
from matplotlib import rcParams


filename = 'log/sne_plain_no_bin.log'

loss = tuple()
align_loss = tuple()
with open(filename, 'r') as f_handler:
    for ln in f_handler:
        ln = ln.strip()
        if 'batch' in ln and 'identity_loss' in ln:
            p = re.compile(r'cur_loss=(\d+.\d+), identity_loss=(\d+.\d+?)')
            match = p.search(ln)
            if match:
                loss += match.group(1),
                align_loss += match.group(2),
                
loss = map(float, loss)
align_loss = map(float, align_loss)

k=60
plt.plot([np.mean(loss[i:i+k]) for i in range(0,len(loss),k)])
# plt.plot(loss)
plt.show()

plt.plot([np.mean(align_loss[i:i+k]) for i in range(0,len(loss),k)])
# plt.plot(align_loss)
plt.show()