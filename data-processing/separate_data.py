
with open('datasets/data/tweets_with_five_emojis.csv', 'r') as all_data:
    all_tweets = all_data.readlines()


train_tweets = all_tweets[:-5000]
with open('datasets/data/train_data_five_emojis.csv', 'w') as train:
    train.writelines(train_tweets)


last_5000_lines = all_tweets[-5000:]
with open('datasets/data/test_data_five_emojis.csv', 'w') as test:
    test.writelines(last_5000_lines)

all_data.close()
train.close()
test.close()