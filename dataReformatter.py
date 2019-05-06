import pandas as pd

def reformat():

    # Read in original data
    df = pd.read_csv('dataset/events.csv', sep=',')

    # Figure out cutoff for what will be test/train data
    mostRecentTimestamp = max(df.timestamp)
    testTrainCutOff = mostRecentTimestamp - 86400000 # 24 hours = 86 400 000 milliseconds

    # Split the data into test and train sets based on timestamp
    train = df[(df['timestamp'] < testTrainCutOff)]
    train['visitorCount'] = train['visitorid'].value_counts()

    trainDf = train[train['visitorCount'] >= 10]
    testDf = df[(df['timestamp'] >= testTrainCutOff)]

    # Use this code if we want separate eventsCounts for each event
    #finalTrainDf = trainDf.groupby(['visitorid','itemid', 'event']).size().reset_index().rename(columns={0:'eventsCount'})
    #refTestDf = testDf.groupby(['visitorid','itemid', 'event']).size().reset_index().rename(columns={0:'eventsCount'})

    # Use this code if we want cumulative eventsCount including all events
    finalTrainDf = trainDf.groupby(['visitorid','itemid']).size().reset_index().rename(columns={0:'eventsCount'})
    refTestDf = testDf.groupby(['visitorid','itemid']).size().reset_index().rename(columns={0:'eventsCount'})

    # Remove the users who only appear in the test data
    interTestDf = refTestDf[refTestDf['visitorid'].isin(finalTrainDf['visitorid'])].reset_index()
    finalTestDf = interTestDf[interTestDf['itemid'].isin(finalTrainDf['itemid'])].reset_index()

    # Save test/train data to CSV fils to be uploaded later
    finalTrainDf.to_csv (r'dataset/trainData.csv', index = None, header=True)
    finalTestDf.to_csv (r'dataset/testData.csv', index = None, header=True)
    
if __name__ == "__main__":
    reformat()
