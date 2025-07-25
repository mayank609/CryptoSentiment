%% Step 1: Load Crypto Tweets CSV
data = readtable('sample_crypto_tweets.csv');
tweets = string(data.Tweet_Text);
btc_price = data.BTC_Price;
dates = data.Date;

%% Handle Small Dataset (Simulate More Rows if Needed)
minRows = 40;
extraRows = minRows - height(data);

if extraRows > 0
    disp('⚠️ Small dataset detected. Simulating additional rows for testing...');
    simulatedTweets = repelem("Simulated Tweet", extraRows)';
    simulatedPrices = repmat(mean(btc_price), extraRows, 1);
    simulatedDates = dates(end) + days(1:extraRows)';

    simulatedData = table(simulatedTweets, simulatedPrices, simulatedDates, ...
        'VariableNames', {'Tweet_Text', 'BTC_Price', 'Date'});

    data = [data; simulatedData];
    tweets = string(data.Tweet_Text);
    btc_price = data.BTC_Price;
    dates = data.Date;
end

%% Step 2: VADER Sentiment
vader_scores = zeros(length(tweets), 1);
for i = 1:length(tweets)
    td = tokenizedDocument(tweets(i));
    td = erasePunctuation(td);
    td = removeStopWords(td);
    vader_scores(i) = vaderSentimentScores(td);
end
data.VADER_Sentiment = vader_scores;
disp('✅ VADER Sentiment Computed');

%% Step 3: Load Sentiment140 Dataset
disp('Loading Sentiment140 dataset...');
sentiment140 = readtable('training.csv', 'ReadVariableNames', false);
labels_raw = sentiment140.Var1;
tweets_raw = sentiment140.Var6;
labels = double(labels_raw == 4);
tweets_train = string(tweets_raw);
disp('✅ Sentiment140 Dataset Loaded');

%% Step 4: Reduce Dataset Size
N = 20000;
idx = randperm(length(tweets_train), N);
tweets_train = tweets_train(idx);
labels = labels(idx);

%% Step 5: Preprocess Training Tweets
documents = tokenizedDocument(tweets_train, 'Language', 'en');
documents = erasePunctuation(documents);
documents = removeStopWords(documents);
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag, 50);
train_features = bag.Counts;
disp('✅ Training Data Prepared');

%% Step 6: Train Naive Bayes
nbModel = fitcnb(train_features, labels);
disp('✅ Naive Bayes Training Completed');

%% Step 7: Apply Naive Bayes to Crypto Tweets
documents_crypto = tokenizedDocument(tweets, 'Language', 'en');
documents_crypto = erasePunctuation(documents_crypto);
documents_crypto = removeStopWords(documents_crypto);
test_features = encode(bag, documents_crypto);
nb_sentiment = predict(nbModel, test_features);
data.NaiveBayes_Sentiment = nb_sentiment;
disp('✅ Crypto Tweets Classified Using Naive Bayes');

%% Step 8: Ratio Rule Sentiment
posWords = tokenizedDocument(["good","great","excellent","positive","love","profit","bullish","increase","rise","up","moon"], 'Language','en');
negWords = tokenizedDocument(["bad","poor","terrible","negative","hate","loss","bearish","decrease","fall","down","dump"], 'Language','en');

ratio_rule_sentiment = zeros(length(tweets), 1);
for i = 1:length(tweets)
    words = tokenizedDocument(tweets(i));
    words = erasePunctuation(words);
    words = removeStopWords(words);
    numPos = sum(ismember(words.Vocabulary, posWords.Vocabulary));
    numNeg = sum(ismember(words.Vocabulary, negWords.Vocabulary));
    if numPos > numNeg
        ratio_rule_sentiment(i) = 1;
    else
        ratio_rule_sentiment(i) = 0;
    end
end
data.RatioRule_Sentiment = ratio_rule_sentiment;
disp('✅ Crypto Tweets Classified Using Ratio Rule');

%% Step 9: Time Series Forecast using ARIMAX
Y = data.BTC_Price;
X = [data.VADER_Sentiment, double(data.NaiveBayes_Sentiment), data.RatioRule_Sentiment];
Y = fillmissing(Y, 'previous');
X = fillmissing(X, 'previous');

model = arima('ARLags',1, 'MALags',1, 'Constant',NaN);

[estModel, estParamCov, logL] = estimate(model, Y, 'X', X);

%% Step 10: Forecast Future BTC Prices
horizon = 10;
XFuture = X(end-horizon+1:end, :);  % Use last available sentiment data
[YForecast, YMSE] = forecast(estModel, horizon, 'Y0', Y, 'X0', X, 'XF', XFuture);

%% Step 11: Plot Forecast
figure;
plot(Y(end-50:end), 'b', 'LineWidth', 1.5);
hold on;
plot(length(Y):(length(Y)+horizon-1), YForecast, 'r--', 'LineWidth', 2);
legend('Actual BTC Price','Forecasted BTC Price');
title('BTC Price Forecast using ARIMAX + Sentiments');
xlabel('Time');
ylabel('BTC Price');
grid on;

%% Step 12: Save Results
writetable(data, 'crypto_tweets_with_sentiment.csv');
disp('✅ Results Saved to crypto_tweets_with_sentiment.csv');
