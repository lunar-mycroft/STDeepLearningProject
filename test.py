import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
from math import log

from tqdm import tqdm

from recomender import Recomender
from util import loadTestData

# Top-k Hit Ratio
# rec is a Recomender object
# data is the entire test dataframe
# k is how many recommendations will be made for each user
# the average hit ratio for all users in the test data is returned
def hitRatio(rec,data,k=0):
    if k == 0:
        k =  len(data.index)
    users = data['visitorid'].unique()
    hitRatios = []
    for user in users:
        recommendations = rec.makeRecomendations(user, k)
        recItems = recommendations['items'].tolist()
        actualItems = data[data['visitorid'] == user].itemid.tolist()
        hits = len(set(recItems) & set(actualItems))
        hitRatios.append(hits / len(data[data['visitorid'] == user].index))

    sumHitRatio = 0
    for hitRatio in hitRatios:
        sumHitRatio += hitRatio
    return sumHitRatio / len(hitRatios)

def nDCG(rec,data,k=0):
    if k == 0:
        k = len(data.index) # It wouldn't do this as a default value
    users = data['visitorid'].unique()
    userNDCGs = []
    for user in users:
        recommendations = rec.makeRecomendations(user, k)
        recItems = recommendations['items'].tolist()
        actualItems = data[data['visitorid'] == user].itemid.tolist()
        hits = set(recItems) & set(actualItems)
        positions = []
        for hit in hits:
            positions.append(recItems.index(hit))
        userNDCG = 0
        for pos in positions:
            userNDCG += 1/(log(pos+2))
        userNDCGs.append(userNDCG)

    sumNDCG = 0
    for nDCG in userNDCGs:
        sumNDCG += nDCG
    return sumNDCG / len(userNDCGs)

# "Main"
if __name__ == "__main__":
    recomender = Recomender('model-60.meta') #TODO: insert model path
    data = loadTestData()
    while True:
        k = input("Please enter k value (or quit to exit): ")
        if str(k).lower() == "quit":
            break
        try:
            k = int(k)
        except:
            print("Please enter an integer")
            continue
        print('Top-k Hit Ratio:', hitRatio(recomender, data[0], int(k)))
        print('nDCG:', nDCG(recomender, data[0], int(k)))
