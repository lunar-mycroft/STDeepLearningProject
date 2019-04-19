#Example code https://medium.com/radon-dev/implicit-bayesian-personalized-ranking-in-tensorflow-b4dfa733c478

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

#---------------------------
# LOAD AND PREPARE THE DATA
#---------------------------

# Load the dataframe from a tab separated file.
df = pd.read_csv('data/usersha1-artmbid-artname-plays.tsv', sep='\t')
    
# Add column names
df = df.drop(df.columns[1], axis=1)
df.columns = ['user', 'artist', 'plays']

# Drop any rows with missing values
df = df.dropna()

# Convert artists names into numerical IDs
df['user_id'] = df['user'].astype("category").cat.codes
df['artist_id'] = df['artist'].astype("category").cat.codes

# Create a lookup frame so we can get the artist
# names back in readable form later.
item_lookup = df[['artist_id', 'artist']].drop_duplicates()
item_lookup['artist_id'] = item_lookup.artist_id.astype(str)

# We drop our old user and artist columns
df = df.drop(['user', 'artist'], axis=1)

# Drop any rows with 0 plays
df = df.loc[df.plays != 0]

# Create lists of all users, artists and plays
users = list(np.sort(df.user_id.unique()))
artists = list(np.sort(df.artist_id.unique()))
plays = list(df.plays)

# Get the rows and columns for our new matrix
rows = df.user_id.astype(float)
cols = df.artist_id.astype(float)

# Contruct a sparse matrix for our users and items containing number of plays
data_sparse = sp.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))

# Get the values of our matrix as a list of user ids
# and item ids. Note that our litsts have the same length
# as each user id repeats one time for each played artist.
uids, iids = data_sparse.nonzero()

#-------------
# HYPERPARAMS
#-------------

epochs = 50
batches = 30
num_factors = 64 # Number of latent features

# Independent lambda regularization values 
# for user, items and bias.
lambda_user = 0.0000001
lambda_item = 0.0000001
lambda_bias = 0.0000001

# Our learning rate 
lr = 0.005

# How many (u,i,j) triplets we sample for each batch
samples = 15000

#-------------------------
# TENSORFLOW GRAPH
#-------------------------

# Set up our Tensorflow graph
graph = tf.Graph()

def init_variable(size, dim, name=None):
    '''
    Helper function to initialize a new variable with
    uniform random values.
    '''
    std = np.sqrt(2 / dim)
    return tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)


def embed(inputs, size, dim, name=None):
    '''
    Helper function to get a Tensorflow variable and create
    an embedding lookup to map our user and item
    indices to vectors.
    '''
    emb = init_variable(size, dim, name)
    return tf.nn.embedding_lookup(emb, inputs)


def get_variable(graph, session, name):
    '''
    Helper function to get the value of a
    Tensorflow variable by name.
    '''
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)
    return v

with graph.as_default():
    '''
    Loss function: 
    -SUM ln σ(xui - xuj) + λ(w1)**2 + λ(w2)**2 + λ(w3)**2 ...
    ln = the natural log
    σ(xuij) = the sigmoid function of xuij.
    λ = lambda regularization value.
    ||W||**2 = the squared L2 norm of our model parameters.
    
    '''

    # Input into our model, in this case our user (u),
    # known item (i) an unknown item (i) triplets.
    u = tf.placeholder(tf.int32, shape=(None, 1))
    i = tf.placeholder(tf.int32, shape=(None, 1))
    j = tf.placeholder(tf.int32, shape=(None, 1))

    # User feature embedding
    u_factors = embed(u, len(users), num_factors, 'user_factors') # U matrix

    # Known and unknown item embeddings
    item_factors = init_variable(len(artists), num_factors, "item_factors") # V matrix
    i_factors = tf.nn.embedding_lookup(item_factors, i)
    j_factors = tf.nn.embedding_lookup(item_factors, j)

    # i and j bias embeddings.
    item_bias = init_variable(len(artists), 1, "item_bias")
    i_bias = tf.nn.embedding_lookup(item_bias, i)
    i_bias = tf.reshape(i_bias, [-1, 1])
    j_bias = tf.nn.embedding_lookup(item_bias, j)
    j_bias = tf.reshape(j_bias, [-1, 1])

    # Calculate the dot product + bias for known and unknown
    # item to get xui and xuj.
    xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
    xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)

    # We calculate xuij.
    xuij = xui - xuj

    # Calculate the mean AUC (area under curve).
    # if xuij is greater than 0, that means that 
    # xui is greater than xuj (and thats what we want).
    u_auc = tf.reduce_mean(tf.to_float(xuij > 0))

    # Output the AUC value to tensorboard for monitoring.
    tf.summary.scalar('auc', u_auc)

    # Calculate the squared L2 norm ||W||**2 multiplied by λ.
    l2_norm = tf.add_n([
        lambda_user * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
        lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
        lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
        lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
        lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias))
        ])

    # Calculate the loss as ||W||**2 - ln σ(Xuij)
    #loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    loss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij))) + l2_norm
    
    # Train using the Adam optimizer to minimize 
    # our loss function.
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    step = opt.minimize(loss)

    # Initialize all tensorflow variables.
    init = tf.global_variables_initializer()

#------------------
# GRAPH EXECUTION
#------------------

# Run the session. 
session = tf.Session(config=None, graph=graph)
session.run(init)

# This has noting to do with tensorflow but gives
# us a nice progress bar for the training.
progress = tqdm(total=batches*epochs)

for _ in range(epochs):
    for _ in range(batches):

        # We want to sample one known and one unknown 
        # item for each user. 

        # First we sample 15000 uniform indices.
        idx = np.random.randint(low=0, high=len(uids), size=samples)

        # We then grab the users matching those indices.
        batch_u = uids[idx].reshape(-1, 1)

        # Then the known items for those users.
        batch_i = iids[idx].reshape(-1, 1)

        # Lastly we randomly sample one unknown item for each user.
        batch_j = np.random.randint(
                low=0, high=len(artists), size=(samples, 1), dtype='int32')

        # Feed our users, known and unknown items to
        # our tensorflow graph. 
        feed_dict = { u: batch_u, i: batch_i, j: batch_j }

        # We run the session.
        _, l, auc = session.run([step, loss, u_auc], feed_dict)

    progress.update(batches)
    progress.set_description('Loss: %.3f | AUC: %.3f' % (l, auc))

progress.close()

#-----------------------
# FIND SIMILAR ARTISTS
#-----------------------

def find_similar_artists(artist=None, num_items=10):
    """Find artists similar to an artist.
    Args:
        artist (str): The name of the artist we want to find similar artists for
        num_items (int): How many similar artists we want to return.
    Returns:
        similar (pandas.DataFrame): DataFrame with num_items artist names and scores
    """

    # Grab our User matrix U
    user_vecs = get_variable(graph, session, 'user_factors')

    # Grab our Item matrix V
    item_vecs = get_variable(graph, session, 'item_factors')

    # Grab our item bias
    item_bi = get_variable(graph, session, 'item_bias').reshape(-1)

    # Get the item id for Lady GaGa
    item_id = int(item_lookup[item_lookup.artist == artist]['artist_id'])

    # Get the item vector for our item_id and transpose it.
    item_vec = item_vecs[item_id].T

    # Calculate the similarity between Lady GaGa and all other artists
    # by multiplying the item vector with our item_matrix
    scores = np.add(item_vecs.dot(item_vec), item_bi).reshape(1,-1)[0]

    # Get the indices for the top 10 scores
    top_10 = np.argsort(scores)[::-1][:num_items]

    # We then use our lookup table to grab the names of these indices
    # and add it along with its score to a pandas dataframe.
    artists, artist_scores = [], []
    
    for idx in top_10:
        artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        artist_scores.append(scores[idx])

    similar = pd.DataFrame({'artist': artists, 'score': artist_scores})

    return similar

print(find_similar_artists(artist='beyoncé'))

#---------------------
# MAKE RECOMMENDATION
#---------------------

def make_recommendation(user_id=None, num_items=10):
    """Recommend items for a given user given a trained model
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        num_items (int): How many recommendations we want to return.
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
    """

    # Grab our user matrix U
    user_vecs = get_variable(graph, session, 'user_factors')

    # Grab our item matrix V
    item_vecs = get_variable(graph, session, 'item_factors')

    # Grab our item bias
    item_bi = get_variable(graph, session, 'item_bias').reshape(-1)

    # Calculate the score for our user for all items. 
    rec_vector = np.add(user_vecs[user_id, :].dot(item_vecs.T), item_bi)

    # Grab the indices of the top users
    item_idx = np.argsort(rec_vector)[::-1][:num_items]

    # Map the indices to artist names and add to dataframe along with scores.
    artists, scores = [], []

    for idx in item_idx:
        artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        scores.append(rec_vector[idx])

    recommendations = pd.DataFrame({'artist': artists, 'score': scores})

    return recommendations

print(make_recommendation(user_id=0))