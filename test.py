import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

from recomender import Recomender

# Top-k Hit Ratio
# rec is a Recomender object
# data is the entire test dataframe
# k is how many recommendations will be made for each user
# the average hit ratio for all users in the test data is returned
def hitRatio(rec,data,k):
    users = data['visitorid'].unique()
    hitRatios = 0
    for user in users:
        recommendations = rec.getRecommendations(user, k)
        hits = data[data['visitorid'] == user & data['itemid'].isin(recommendations)]
        hitRatios.append(len(hits.index) / len(data['visitorid'] == user))

    sumHitRatio = 0
    for hitRatio in hitRatios:
        sumHitRatio += hitRatio
    return sumHitRatio / len(hitRatios.index)

def nDCG(rec,data):
