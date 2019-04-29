import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

from util import init_variable, embed, get_variable, loadModal, loadTestData, preprocessTestData

class Recomender():
    def __init__(self,modelPath):
        self.df, self.users, self.items, self.numEvents, self = preprocessTestData(loadTestData())
        self.graph, self.session = loadModal(modelPath)

    def makeRecomendations(self,visitorid, numRecs):

        user_vecs = get_variable(self.graph, self.session, 'user_factors') #U matrix
        item_vecs = get_variable(graph, session, 'item_factors') #V matrix
        item_bi = get_variable(graph, session, 'item_bias').reshape(-1) # Baises

        
        rec_vector = np.add(user_vecs[user, :].dot(item_vecs.T), item_bi) # Calculate score for all items for the given user

        # Grab the indices of the top users
        item_idx = np.argsort(rec_vector)[::-1][:n]

        # Map the indices to artist names and add to dataframe along with scores.
        artists, scores = [], []

        for idx in item_idx:
            artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
            scores.append(rec_vector[idx])

        recommendations = pd.DataFrame({'artist': artists, 'score': scores})

        return recommendations
    def getScore(self,visitorid, itemid):
        

        

