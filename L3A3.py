#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('embeddingsdata.xlsx')

vector1 = np.array([df['embed_1'][8], df['embed_2'][8]])
vector2 = np.array([df['embed_1'][10], df['embed_2'][10]])

r_values = np.arange(1, 11)

distances = [np.sum(np.abs(vector1 - vector2)**r)**(1/r) for r in r_values]

plt.plot(r_values, distances, marker='o', color='y')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.grid(True)
plt.show()

