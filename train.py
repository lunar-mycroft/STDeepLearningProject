import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

from dataReformatter import reformat


#Helper functions from example

def init_variable(size, dim, name=None):
    std = np.sqrt(2 / dim)
    return tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)


def embed(inputs, size, dim, name=None):
    emb = init_variable(size, dim, name)
    return tf.nn.embedding_lookup(emb, inputs)


def get_variable(graph, session, name):
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)
    return v

# Loading and preprocessing functions

def loadData():
    while True:
        try:
            res = pd.read_csv('dataset/trainData.csv', sep=',')
            break
        except:
            print("reformatting data")
            reformat()
    return res

def preprocessTrainingData(df):
    df = df.dropna()

    df['user_id'] = df['visitorid'].astype("category").cat.codes
    df['item_id'] = df['itemid'].astype("category").cat.codes

    # Create a lookup frame so we can get the artist
    # names back in readable form later.
    item_lookup = df[['user_id', 'artist']].drop_duplicates()
    item_lookup['user_id'] = item_lookup.visitorid.astype(str)

    df = df.loc[df.count != 0]

    users = list(np.sort(df.user_id.unique()))
    items = list(np.sort(df.item_id.unique()))
    numEvents = list(df.count)

    return df, users, items, numEvents

def makeMatrix(tup):
    df, users, items, numEvents = tup

    rows = df.user_id.astype(float)
    cols = df.item_id.astype(float)

    data_sparse = sp.csr_matrix((numEvents, (rows, cols)), shape=(len(users), len(items)))

    uids, iids = data_sparse.nonzero()

    return df, users, items, numEvents, rows, cols, data_sparse, uids, iids


def train(epochs = 50, batches = 30, num_factors = 64):
    rawData = loadData()
    preprocessed = preprocessTrainingData(rawData)
    df, users, items, numEvents, rows, cols, data_sparse, uids, iids = makeMatrix(preprocessed)

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


    with graph.as_default():

        # Input into our model, in this case our user (u),
        # known item (i) an unknown item (i) triplets.
        u = tf.placeholder(tf.int32, shape=(None, 1))
        i = tf.placeholder(tf.int32, shape=(None, 1))
        j = tf.placeholder(tf.int32, shape=(None, 1))

        # User feature embedding
        u_factors = embed(u, len(users), num_factors, 'user_factors') # U matrix

        # Known and unknown item embeddings
        item_factors = init_variable(len(items), num_factors, "item_factors") # V matrix
        i_factors = tf.nn.embedding_lookup(item_factors, i)
        j_factors = tf.nn.embedding_lookup(item_factors, j)

        # i and j bias embeddings.
        item_bias = init_variable(len(items), 1, "item_bias")
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

    #Start the saver
    saver = tf.train.Saver()

    # Run the session.
    session = tf.Session(config=None, graph=graph)
    session.run(init)
    saver.save(session, 'model')

    # This has noting to do with tensorflow but gives
    # us a nice progress bar for the training.
    progress = tqdm(total=batches*epochs)



    for i in range(epochs):
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
                    low=0, high=len(items), size=(samples, 1), dtype='int32')

            # Feed our users, known and unknown items to
            # our tensorflow graph.
            feed_dict = { u: batch_u, i: batch_i, j: batch_j }

            # We run the session.
            _, l, auc = session.run([step, loss, u_auc], feed_dict)

        saver.save(session, 'model',global_step=i*batches)

        progress.update(batches)
        progress.set_description('Loss: %.3f | AUC: %.3f' % (l, auc))


    progress.close()

if __name__ == "__main__":
    train()
