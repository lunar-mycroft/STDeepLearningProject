import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

from util import init_variable, embed, get_variable, loadModal, loadTestData, preprocessTestData

class Recomender():
    def __init__(self,modelPath):
        self.df, self.users, self.items, self.numEvents, self.lookUpUser, self.lookUpItem = preprocessTestData(loadTestData())
        self.lookUpUserRev = {v: k for k, v in self.lookUpUser.items()}
        self.lookUpItemRev = {v: k for k, v in self.lookUpItem.items()}
        self.graph, self.session = loadModal(modelPath)

    def makeRecomendations(self,visitorid, numRecs):
        if not self.hasVisitor(visitorid):
            raise ValueError("visitor does not exist")

        user = self.lookUpUser[visitorid]

        user_vecs = get_variable(self.graph, self.session, 'user_factors') #U matrix
        item_vecs = get_variable(self.graph, self.session, 'item_factors') #V matrix
        item_bi = get_variable(self.graph, self.session, 'item_bias').reshape(-1) # Baises

        
        rec_vector = np.add(user_vecs[user, :].dot(item_vecs.T), item_bi) # Calculate score for all items for the given user

        item_idx = np.argsort(rec_vector)[::-1][:numRecs]

        # Map the indices to artist names and add to dataframe along with scores.
        items, scores = [self.lookUpItemRev[idx] for idx in item_idx], [rec_vector[idx] for idx in item_idx]

        for idx in item_idx:
            items.append(self.lookUpItemRev[idx])
            scores.append(rec_vector[idx])

        recommendations = pd.DataFrame({'items': items, 'score': scores})

        return recommendations
    def getScore(self,visitorid, itemid):
        if not (self.hasItem(itemid) and self.hasVisitor(visitorid)):
            raise ValueError("User or item did not exist")
        user = self.lookUpUser[visitorid]
        item = self.lookUpItem[itemid]

        user_vecs = get_variable(self.graph, self.session, 'user_factors') #U matrix
        item_vecs = get_variable(self.graph, self.session, 'item_factors') #V matrix
        item_bi = get_variable(self.graph, self.session, 'item_bias').reshape(-1) # Baises

        rec_vector = np.add(user_vecs[user, :].dot(item_vecs.T), item_bi) # Calculate score for all items for the given user

        return rec_vector[item]

    def hasVisitor(self, visitorid):
        return visitorid in self.lookUpUser

    def hasItem(self, itemid):
        return itemid in self.lookUpItem

    def items(self):
        for itemid in self.lookUpItem:
            yield itemid

    def visitors(self):
        for visitorid in self.lookUpUser:
            yield visitorid        

        

