#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:14:20 2018

@author: nicob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:35:40 2018

@author: nicob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:49:47 2018

@author: nicob
"""
import matplotlib.pyplot as plt
import time
plt.style.use('seaborn')
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing



