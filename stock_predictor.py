# Final code as of May 1, 2025
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, LeakyReLU, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from pickle import load, dump
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import warnings
import nltk

# --- Main Script ---
def main():
    warnings.filterwarnings("ignore")
    
    # Ensure NLTK data is available
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    stock_name = 'TSLA'

    # --- Sentiment Analysis ---
    print("--- Part 1: Sentiment Analysis ---")
    try:
        all_tweets = pd.read_csv('stock_tweets.csv')
        df = all_tweets[all_tweets['Stock Name'] == 'TSLA']
        sent_df = df.copy()
        sent_df["sentiment_score"] = 0.0
        sent_df["Negative"] = 0.0
        sent_df["Neutral"] = 0.0
        sent_df["Positive"] = 0.0
        
        pd.DataFrame.iteritems = pd.DataFrame.items
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Performing sentiment analysis on tweets...")
        for indx, row in tqdm(sent_df.T.iteritems(), total=len(sent_df)):
            try:
                sentence_i = unicodedata.normalize('NFKD', sent_df.loc[indx, 'Tweet'])
                sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
                sent_df.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
                sent_df.at[indx, 'Negative'] = sentence_sentiment['neg']
                sent_df.at[indx, 'Neutral'] = sentence_sentiment['neu']
                sent_df.at[indx, 'Positive'] = sentence_sentiment['pos']
            except TypeError:
                print(f"Skipping row {indx} due to TypeError.")
        
        sent_df['Date'] = pd.to_datetime(sent_df['Date']).dt.date
        sent_df = sent_df.drop(columns=['Stock Name', 'Company Name'], errors='ignore')
        twitter_df = sent_df.groupby([sent_df['Date']]).mean(numeric_only=True)
    except FileNotFoundError:
        print("Warning: 'stock_tweets.csv' not found. Proceeding without sentiment data.")
        twitter_df = pd.DataFrame()

    # --- Data Merging & Feature Engineering ---
    print("\n--- Part 2: Data Merging & Feature Engineering ---")
    try:
        all_stocks = pd.read_csv('stock_yfinance_data.csv')
        stock_df = all_stocks[all_stocks['Stock Name'] == "TSLA"]
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
        final_df = stock_df.join(twitter_df, how="left", on="Date")
        final_df = final_df.drop(columns=['Stock Name'])
        final_df.drop(columns=['Negative', 'Neutral', 'Positive'], inplace=True, errors='ignore')
        final_df = final_df.ffill().bfill() # Fill any NaNs from merging
    except FileNotFoundError:
        print("Error: 'stock_yfinance_data.csv' not found. Cannot proceed.")
        return

    def get_tech_ind(data):
        data['MA7'] = data['Close'].rolling(window=7).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MACD'] = data['Close'].ewm(span=26).mean() - data['Open'].ewm(span=12,adjust=False).mean()
        data['20SD'] = data['Close'].rolling(20).std()
        data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
        data['lower_band'] = data['MA20'] - (data['20SD'] * 2)
        data['EMA'] = data['Close'].ewm(com=0.5).mean()
        data['logmomentum'] = np.log(data['Close'].replace(0, 1e-9) - 1)
        return data

    tech_df = get_tech_ind(final_df)
    dataset = tech_df.iloc[20:,:].reset_index(drop=True).ffill()
    
    datetime_series = pd.to_datetime(dataset['Date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index).sort_values(by='Date').drop(columns='Date')

    # --- Preprocessing & Batching ---
    print("\n--- Part 3: Preprocessing & Batching ---")
    def normalize_data(df, feature_range, target_column):
        target_df_series = pd.DataFrame(df[target_column])
        data = pd.DataFrame(df.iloc[:, :])
        X_scaler = MinMaxScaler(feature_range=feature_range)
        y_scaler = MinMaxScaler(feature_range=feature_range)
        X_scale_dataset = X_scaler.fit_transform(data)
        y_scale_dataset = y_scaler.fit_transform(target_df_series)
        dump(X_scaler, open('X_scaler.pkl', 'wb'))
        dump(y_scaler, open('y_scaler.pkl', 'wb'))
        return X_scale_dataset, y_scale_dataset

    def batch_data(x_data, y_data, batch_size, predict_period):
        X_batched, y_batched, yc = [], [], []
        for i in range(len(x_data) - batch_size - predict_period):
            X_batched.append(x_data[i: i + batch_size])
            y_batched.append(y_data[i + batch_size: i + batch_size + predict_period, 0])
            yc.append(y_data[i: i + batch_size])
        return np.array(X_batched), np.array(y_batched), np.array(yc)

    def split_train_test(data):
        train_size = len(data) - 20
        return data[:train_size], data[train_size:]

    def predict_index(dataset, X_train, batch_size):
        train_predict_index = dataset.iloc[batch_size: X_train.shape[0] + batch_size, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] + batch_size:, :].index
        return train_predict_index, test_predict_index

    X_scale_dataset, y_scale_dataset = normalize_data(dataset, (-1,1), "Close")
    X_batched, y_batched, yc = batch_data(X_scale_dataset, y_scale_dataset, batch_size=5, predict_period=1)
    X_train, X_test = split_train_test(X_batched)
    y_train, y_test = split_train_test(y_batched)
    yc_train, yc_test = split_train_test(yc)
    index_train, index_test = predict_index(dataset, X_train, 5)

    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    # --- Model Architecture ---
    print("\n--- Part 4: Model Architecture ---")
    def make_generator_model():
        return Sequential([LSTM(1024, return_sequences=True, input_shape=(input_dim, feature_size), recurrent_dropout=0.3),
                           LSTM(512, return_sequences=True, recurrent_dropout=0.3),
                           LSTM(256, return_sequences=True, recurrent_dropout=0.3),
                           LSTM(128, return_sequences=True, recurrent_dropout=0.3),
                           LSTM(64, recurrent_dropout=0.3),
                           Dense(32), Dense(16), Dense(8), Dense(units=output_dim)])

    def make_discriminator_model():
        return Sequential([Conv1D(8, input_shape=(input_dim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
                           Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
                           Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
                           Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
                           Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
                           LeakyReLU(), Dense(220, use_bias=False), LeakyReLU(),
                           Dense(220, use_bias=False, activation='relu'), Dense(1, activation='sigmoid')])

    # --- Training ---
    print("\n--- Part 5: GAN Training ---")
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def discriminator_loss(real_output, fake_output):
        return loss_fn(tf.ones_like(real_output), real_output) + loss_fn(tf.zeros_like(fake_output), fake_output)
    def generator_loss(fake_output):
        return loss_fn(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(real_x, real_y, yc, generator, discriminator, g_opt, d_opt):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(real_x, training=True)
            d_fake_input = tf.concat([tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1]), yc], axis=1)
            d_real_input = tf.concat([tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1]), yc], axis=1)
            real_output, fake_output = discriminator(d_real_input, training=True), discriminator(d_fake_input, training=True)
            g_loss, disc_loss = generator_loss(fake_output), discriminator_loss(real_output, fake_output)
        g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        d_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    def train_model(epochs=50, checkpoint=25):
        generator, discriminator = make_generator_model(), make_discriminator_model()
        g_optimizer = tf.keras.optimizers.Adam(5e-4)
        d_optimizer = tf.keras.optimizers.Adam(5e-4)
        os.makedirs(f'./models_gan/{stock_name}', exist_ok=True)
        print(f"Starting training for {epochs} epochs...")
        for epoch in tqdm(range(epochs)):
            train_step(X_train, y_train, yc_train, generator, discriminator, g_optimizer, d_optimizer)
            if (epoch + 1) % checkpoint == 0:
                generator.save(f'./models_gan/{stock_name}/generator_V_{epoch+1}.h5')
        print("Training complete.")
        return generator

    # Execute training
    generator = train_model(epochs=50)

    # --- Evaluation ---
    print("\n--- Part 6: Evaluation ---")
    @tf.function
    def eval_op(gen, real_x):
        return gen(real_x, training=False)
        
    def plot_test_data(real_price, predicted_price, test_index):
        y_scaler = load(open('y_scaler.pkl', 'rb'))
        rescaled_real = y_scaler.inverse_transform(real_price)
        rescaled_predicted = y_scaler.inverse_transform(predicted_price)
        
        real_df = pd.DataFrame(rescaled_real.flatten(), index=test_index[:len(rescaled_real.flatten())], columns=["Real Price"])
        pred_df = pd.DataFrame(rescaled_predicted.flatten(), index=test_index[:len(rescaled_predicted.flatten())], columns=["Predicted Price"])
        
        plt.figure(figsize=(16, 8))
        plt.plot(real_df["Real Price"], color='#00008B', label="Real Price")
        plt.plot(pred_df["Predicted Price"], color='#8B0000', linestyle='--', label="Predicted Price")
        plt.title(f"Prediction on Test Data for {stock_name}", fontsize=20)
        plt.xlabel("Date")
        plt.ylabel("Stock Price (USD)")
        plt.legend()
        plt.savefig(f"{stock_name}_test_prediction.png")
        print(f"Test prediction plot saved to {stock_name}_test_prediction.png")

    predicted_test_data = eval_op(generator, X_test)
    plot_test_data(y_test, predicted_test_data, index_test)

if __name__ == '__main__':
    main()
# Simulated code change on 2025-02-01T23:08:29
# Simulated code change on 2025-02-01T20:18:01
# Simulated code change on 2025-02-01T19:20:03
# Simulated code change on 2025-02-01T10:12:45
# Simulated code change on 2025-02-01T09:00:58
# Simulated code change on 2025-02-01T23:07:45
# Simulated code change on 2025-02-01T23:29:19
# Simulated code change on 2025-02-01T11:07:49
# Simulated code change on 2025-02-01T14:38:12
# Simulated code change on 2025-02-01T12:23:02
# Simulated code change on 2025-02-02T22:59:43
# Simulated code change on 2025-02-02T12:41:35
# Simulated code change on 2025-02-02T14:32:05
# Simulated code change on 2025-02-02T09:03:19
# Simulated code change on 2025-02-02T18:45:12
# Simulated code change on 2025-02-02T09:52:23
# Simulated code change on 2025-02-02T10:44:27
# Simulated code change on 2025-02-02T10:24:51
# Simulated code change on 2025-02-02T13:18:59
# Simulated code change on 2025-02-02T13:32:59
# Simulated code change on 2025-02-02T22:26:55
# Simulated code change on 2025-02-02T18:05:37
# Simulated code change on 2025-02-03T11:59:57
# Simulated code change on 2025-02-03T14:14:45
# Simulated code change on 2025-02-03T11:39:42
# Simulated code change on 2025-02-03T16:31:44
# Simulated code change on 2025-02-03T23:08:28
# Simulated code change on 2025-02-03T12:55:54
# Simulated code change on 2025-02-03T17:42:17
# Simulated code change on 2025-02-03T16:46:13
# Simulated code change on 2025-02-03T19:47:11
# Simulated code change on 2025-02-03T17:09:12
# Simulated code change on 2025-02-06T12:12:51
# Simulated code change on 2025-02-06T13:31:21
# Simulated code change on 2025-02-06T16:45:27
# Simulated code change on 2025-02-06T17:48:57
# Simulated code change on 2025-02-06T15:39:17
# Simulated code change on 2025-02-06T16:28:18
# Simulated code change on 2025-02-06T12:12:10
# Simulated code change on 2025-02-06T22:50:26
# Simulated code change on 2025-02-06T09:10:26
# Simulated code change on 2025-02-06T15:59:22
# Simulated code change on 2025-02-06T22:48:46
# Simulated code change on 2025-02-06T11:25:31
# Simulated code change on 2025-02-06T13:49:50
# Simulated code change on 2025-02-06T21:08:28
# Simulated code change on 2025-02-06T13:19:34
# Simulated code change on 2025-02-08T16:47:38
# Simulated code change on 2025-02-08T23:38:59
# Simulated code change on 2025-02-08T11:02:42
# Simulated code change on 2025-02-08T17:54:09
# Simulated code change on 2025-02-08T09:04:06
# Simulated code change on 2025-02-08T15:54:51
# Simulated code change on 2025-02-09T22:06:22
# Simulated code change on 2025-02-09T11:38:07
# Simulated code change on 2025-02-09T18:42:18
# Simulated code change on 2025-02-09T10:11:57
# Simulated code change on 2025-02-09T22:06:56
# Simulated code change on 2025-02-09T16:19:58
# Simulated code change on 2025-02-09T13:08:18
# Simulated code change on 2025-02-09T10:55:57
# Simulated code change on 2025-02-09T17:48:24
# Simulated code change on 2025-02-09T12:13:22
# Simulated code change on 2025-02-09T10:39:42
# Simulated code change on 2025-02-09T21:49:58
# Simulated code change on 2025-02-09T13:10:50
# Simulated code change on 2025-02-09T16:31:48
# Simulated code change on 2025-02-09T14:56:48
# Simulated code change on 2025-02-10T19:12:17
# Simulated code change on 2025-02-10T22:04:00
# Simulated code change on 2025-02-10T10:46:58
# Simulated code change on 2025-02-10T09:21:21
# Simulated code change on 2025-02-10T23:47:31
# Simulated code change on 2025-02-10T16:26:16
# Simulated code change on 2025-02-10T12:02:01
# Simulated code change on 2025-02-10T15:51:46
# Simulated code change on 2025-02-10T10:32:51
# Simulated code change on 2025-02-10T09:07:06
# Simulated code change on 2025-02-10T16:18:42
# Simulated code change on 2025-02-10T15:14:23
# Simulated code change on 2025-02-10T09:49:45
# Simulated code change on 2025-02-11T13:26:37
# Simulated code change on 2025-02-11T14:12:41
# Simulated code change on 2025-02-11T10:03:21
# Simulated code change on 2025-02-11T19:43:54
# Simulated code change on 2025-02-11T22:36:46
# Simulated code change on 2025-02-11T21:11:54
# Simulated code change on 2025-02-11T12:03:19
# Simulated code change on 2025-02-11T12:20:41
# Simulated code change on 2025-02-11T13:45:29
# Simulated code change on 2025-02-11T22:08:23
# Simulated code change on 2025-02-11T16:52:57
# Simulated code change on 2025-02-12T21:21:43
# Simulated code change on 2025-02-12T15:18:44
# Simulated code change on 2025-02-12T21:30:35
# Simulated code change on 2025-02-12T20:31:33
# Simulated code change on 2025-02-12T13:56:24
# Simulated code change on 2025-02-12T16:43:30
# Simulated code change on 2025-02-13T17:44:57
# Simulated code change on 2025-02-13T11:34:48
# Simulated code change on 2025-02-13T15:41:50
# Simulated code change on 2025-02-13T22:17:29
# Simulated code change on 2025-02-13T23:49:11
# Simulated code change on 2025-02-13T22:49:20
# Simulated code change on 2025-02-13T13:06:00
# Simulated code change on 2025-02-13T11:50:16
# Simulated code change on 2025-02-13T21:33:14
# Simulated code change on 2025-02-13T15:01:45
# Simulated code change on 2025-02-13T18:52:10
# Simulated code change on 2025-02-13T22:57:25
# Simulated code change on 2025-02-13T10:50:26
# Simulated code change on 2025-02-13T14:21:48
# Simulated code change on 2025-02-13T09:02:52
# Simulated code change on 2025-02-14T14:52:17
# Simulated code change on 2025-02-14T15:43:13
# Simulated code change on 2025-02-14T09:28:58
# Simulated code change on 2025-02-14T12:57:59
# Simulated code change on 2025-02-14T14:19:56
# Simulated code change on 2025-02-14T13:13:02
# Simulated code change on 2025-02-14T12:25:22
# Simulated code change on 2025-02-14T09:50:08
# Simulated code change on 2025-02-14T16:23:14
# Simulated code change on 2025-02-14T19:14:54
# Simulated code change on 2025-02-14T22:16:14
# Simulated code change on 2025-02-14T21:48:28
# Simulated code change on 2025-02-14T10:57:19
# Simulated code change on 2025-02-15T22:15:53
# Simulated code change on 2025-02-15T14:24:34
# Simulated code change on 2025-02-15T16:53:45
# Simulated code change on 2025-02-15T22:16:51
# Simulated code change on 2025-02-15T12:21:08
# Simulated code change on 2025-02-15T22:02:56
# Simulated code change on 2025-02-15T22:28:43
# Simulated code change on 2025-02-15T16:17:22
# Simulated code change on 2025-02-16T22:51:32
# Simulated code change on 2025-02-16T17:00:11
# Simulated code change on 2025-02-16T21:16:19
# Simulated code change on 2025-02-16T14:37:49
# Simulated code change on 2025-02-16T11:51:40
# Simulated code change on 2025-02-17T17:19:10
# Simulated code change on 2025-02-17T11:14:58
# Simulated code change on 2025-02-17T15:17:25
# Simulated code change on 2025-02-17T13:54:40
# Simulated code change on 2025-02-17T16:31:31
# Simulated code change on 2025-02-17T22:38:37
# Simulated code change on 2025-02-17T09:08:45
# Simulated code change on 2025-02-17T22:10:09
# Simulated code change on 2025-02-17T10:42:43
# Simulated code change on 2025-02-17T12:55:32
# Simulated code change on 2025-02-17T09:59:43
# Simulated code change on 2025-02-17T14:22:56
# Simulated code change on 2025-02-17T19:37:25
# Simulated code change on 2025-02-17T22:47:09
# Simulated code change on 2025-02-17T12:43:24
# Simulated code change on 2025-02-18T13:27:09
# Simulated code change on 2025-02-18T21:13:43
# Simulated code change on 2025-02-18T12:15:10
# Simulated code change on 2025-02-18T16:51:15
# Simulated code change on 2025-02-18T15:50:20
# Simulated code change on 2025-02-18T21:26:15
# Simulated code change on 2025-02-18T22:32:34
# Simulated code change on 2025-02-18T09:11:32
# Simulated code change on 2025-02-18T22:00:50
# Simulated code change on 2025-02-18T23:28:12
# Simulated code change on 2025-02-18T18:16:06
# Simulated code change on 2025-02-19T10:59:30
# Simulated code change on 2025-02-19T19:17:02
# Simulated code change on 2025-02-19T18:06:42
# Simulated code change on 2025-02-19T12:15:32
# Simulated code change on 2025-02-19T15:29:01
# Simulated code change on 2025-02-19T10:40:59
# Simulated code change on 2025-02-19T19:29:08
# Simulated code change on 2025-02-20T18:28:35
# Simulated code change on 2025-02-20T11:28:24
# Simulated code change on 2025-02-20T13:52:14
# Simulated code change on 2025-02-20T14:55:50
# Simulated code change on 2025-02-20T21:40:25
# Simulated code change on 2025-02-20T16:37:00
# Simulated code change on 2025-02-20T14:22:12
# Simulated code change on 2025-02-20T11:24:23
# Simulated code change on 2025-02-20T13:50:18
# Simulated code change on 2025-02-20T16:15:17
# Simulated code change on 2025-02-20T14:02:16
# Simulated code change on 2025-02-20T14:26:27
# Simulated code change on 2025-02-20T09:04:05
# Simulated code change on 2025-02-21T23:16:51
# Simulated code change on 2025-02-21T17:13:08
# Simulated code change on 2025-02-21T13:16:30
# Simulated code change on 2025-02-21T11:57:19
# Simulated code change on 2025-02-21T21:33:22
# Simulated code change on 2025-02-21T21:20:50
# Simulated code change on 2025-02-21T13:44:51
# Simulated code change on 2025-02-22T14:06:54
# Simulated code change on 2025-02-22T14:23:12
# Simulated code change on 2025-02-22T19:40:38
# Simulated code change on 2025-02-22T16:11:17
# Simulated code change on 2025-02-22T10:55:32
# Simulated code change on 2025-02-22T10:53:35
# Simulated code change on 2025-02-22T09:00:17
# Simulated code change on 2025-02-22T14:11:00
# Simulated code change on 2025-02-22T11:49:19
# Simulated code change on 2025-02-22T16:06:45
# Simulated code change on 2025-02-22T13:47:22
# Simulated code change on 2025-02-22T12:30:26
# Simulated code change on 2025-02-22T16:03:49
# Simulated code change on 2025-02-23T22:56:17
# Simulated code change on 2025-02-23T21:25:58
# Simulated code change on 2025-02-23T21:51:56
# Simulated code change on 2025-02-23T22:46:18
# Simulated code change on 2025-02-23T15:21:03
# Simulated code change on 2025-02-23T19:00:11
# Simulated code change on 2025-02-23T23:05:33
# Simulated code change on 2025-02-23T10:07:08
# Simulated code change on 2025-02-23T22:01:40
# Simulated code change on 2025-02-23T22:16:20
# Simulated code change on 2025-02-23T20:01:39
# Simulated code change on 2025-02-24T19:40:58
# Simulated code change on 2025-02-24T21:34:02
# Simulated code change on 2025-02-24T15:17:49
# Simulated code change on 2025-02-24T18:55:29
# Simulated code change on 2025-02-24T23:51:27
# Simulated code change on 2025-02-24T20:44:08
# Simulated code change on 2025-02-24T17:24:59
# Simulated code change on 2025-02-24T14:59:00
# Simulated code change on 2025-02-24T15:50:40
# Simulated code change on 2025-02-24T16:23:42
# Simulated code change on 2025-02-24T09:50:24
# Simulated code change on 2025-02-24T16:09:18
# Simulated code change on 2025-02-24T21:01:16
# Simulated code change on 2025-02-24T10:28:17
# Simulated code change on 2025-02-25T17:51:16
# Simulated code change on 2025-02-25T21:33:44
# Simulated code change on 2025-02-25T19:51:52
# Simulated code change on 2025-02-25T20:30:32
# Simulated code change on 2025-02-25T12:36:40
# Simulated code change on 2025-02-25T09:50:55
# Simulated code change on 2025-02-25T15:57:23
# Simulated code change on 2025-02-25T17:16:59
# Simulated code change on 2025-02-25T14:07:01
# Simulated code change on 2025-02-25T13:53:24
# Simulated code change on 2025-02-25T20:35:46
# Simulated code change on 2025-02-25T19:35:48
# Simulated code change on 2025-02-25T22:09:20
# Simulated code change on 2025-02-25T22:47:23
# Simulated code change on 2025-02-26T14:50:00
# Simulated code change on 2025-02-26T17:15:59
# Simulated code change on 2025-02-26T19:45:52
# Simulated code change on 2025-02-26T18:02:18
# Simulated code change on 2025-02-26T11:04:36
# Simulated code change on 2025-02-26T19:25:28
# Simulated code change on 2025-02-26T13:38:40
# Simulated code change on 2025-02-28T09:24:17
# Simulated code change on 2025-02-28T22:22:38
# Simulated code change on 2025-02-28T23:13:43
# Simulated code change on 2025-02-28T16:46:58
# Simulated code change on 2025-02-28T16:56:13
# Simulated code change on 2025-02-28T11:48:41
# Simulated code change on 2025-02-28T18:31:05
# Simulated code change on 2025-02-28T23:54:03
# Simulated code change on 2025-02-28T22:02:50
# Simulated code change on 2025-02-28T09:44:04
# Simulated code change on 2025-02-28T10:19:39
# Simulated code change on 2025-02-28T12:24:56
# Simulated code change on 2025-03-01T17:16:43
# Simulated code change on 2025-03-01T21:17:37
# Simulated code change on 2025-03-01T19:20:19
# Simulated code change on 2025-03-01T12:17:55
# Simulated code change on 2025-03-01T13:12:00
# Simulated code change on 2025-03-01T18:27:35
# Simulated code change on 2025-03-01T16:08:14
# Simulated code change on 2025-03-01T22:19:57
# Simulated code change on 2025-03-01T22:47:43
# Simulated code change on 2025-03-01T16:53:02
# Simulated code change on 2025-03-02T19:36:06
# Simulated code change on 2025-03-02T18:03:20
# Simulated code change on 2025-03-02T18:20:15
# Simulated code change on 2025-03-02T17:28:41
# Simulated code change on 2025-03-02T18:08:20
# Simulated code change on 2025-03-03T19:32:14
# Simulated code change on 2025-03-03T14:35:35
# Simulated code change on 2025-03-03T12:30:38
# Simulated code change on 2025-03-03T16:24:06
# Simulated code change on 2025-03-03T20:10:38
# Simulated code change on 2025-03-03T14:49:53
# Simulated code change on 2025-03-03T12:35:52
# Simulated code change on 2025-03-03T21:35:52
# Simulated code change on 2025-03-03T11:52:37
# Simulated code change on 2025-03-03T19:28:42
# Simulated code change on 2025-03-03T20:08:40
# Simulated code change on 2025-03-03T12:26:30
# Simulated code change on 2025-03-03T23:19:04
# Simulated code change on 2025-03-05T09:47:58
# Simulated code change on 2025-03-05T12:50:50
# Simulated code change on 2025-03-05T15:47:41
# Simulated code change on 2025-03-05T20:20:17
# Simulated code change on 2025-03-05T21:52:20
# Simulated code change on 2025-03-05T23:10:32
# Simulated code change on 2025-03-05T09:07:41
# Simulated code change on 2025-03-05T15:36:37
# Simulated code change on 2025-03-05T20:10:24
# Simulated code change on 2025-03-05T13:04:23
# Simulated code change on 2025-03-05T16:35:19
# Simulated code change on 2025-03-05T21:54:29
# Simulated code change on 2025-03-06T18:57:54
# Simulated code change on 2025-03-06T17:45:57
# Simulated code change on 2025-03-06T17:02:24
# Simulated code change on 2025-03-06T17:58:11
# Simulated code change on 2025-03-06T12:10:44
# Simulated code change on 2025-03-06T18:44:20
# Simulated code change on 2025-03-06T10:59:16
# Simulated code change on 2025-03-06T13:48:43
# Simulated code change on 2025-03-06T23:49:38
# Simulated code change on 2025-03-06T22:08:34
# Simulated code change on 2025-03-06T17:02:24
# Simulated code change on 2025-03-06T18:10:08
# Simulated code change on 2025-03-06T12:22:35
# Simulated code change on 2025-03-06T15:53:54
# Simulated code change on 2025-03-07T11:44:49
# Simulated code change on 2025-03-07T20:58:06
# Simulated code change on 2025-03-07T12:15:46
# Simulated code change on 2025-03-07T22:31:54
# Simulated code change on 2025-03-07T20:32:32
# Simulated code change on 2025-03-07T21:32:23
# Simulated code change on 2025-03-07T11:27:16
# Simulated code change on 2025-03-07T22:18:02
# Simulated code change on 2025-03-07T20:08:33
# Simulated code change on 2025-03-07T23:56:24
# Simulated code change on 2025-03-07T18:36:42
# Simulated code change on 2025-03-08T19:20:03
# Simulated code change on 2025-03-08T13:39:02
# Simulated code change on 2025-03-08T14:38:09
# Simulated code change on 2025-03-08T16:43:22
# Simulated code change on 2025-03-08T10:53:00
# Simulated code change on 2025-03-08T19:21:22
# Simulated code change on 2025-03-08T16:15:02
# Simulated code change on 2025-03-08T17:05:26
# Simulated code change on 2025-03-08T14:08:02
# Simulated code change on 2025-03-08T12:02:30
# Simulated code change on 2025-03-08T21:38:38
# Simulated code change on 2025-03-08T23:24:17
# Simulated code change on 2025-03-08T16:36:17
# Simulated code change on 2025-03-08T11:30:06
# Simulated code change on 2025-03-08T17:55:43
# Simulated code change on 2025-03-09T16:48:30
# Simulated code change on 2025-03-09T12:43:20
# Simulated code change on 2025-03-09T12:40:00
# Simulated code change on 2025-03-09T17:47:54
# Simulated code change on 2025-03-09T11:02:02
# Simulated code change on 2025-03-09T18:18:35
# Simulated code change on 2025-03-09T11:37:53
# Simulated code change on 2025-03-09T22:27:59
# Simulated code change on 2025-03-09T23:09:33
# Simulated code change on 2025-03-09T09:03:02
# Simulated code change on 2025-03-09T09:55:24
# Simulated code change on 2025-03-09T19:13:43
# Simulated code change on 2025-03-09T13:40:16
# Simulated code change on 2025-03-10T19:32:30
# Simulated code change on 2025-03-10T09:57:29
# Simulated code change on 2025-03-10T20:25:06
# Simulated code change on 2025-03-10T09:18:00
# Simulated code change on 2025-03-10T15:52:43
# Simulated code change on 2025-03-10T13:26:02
# Simulated code change on 2025-03-10T12:59:13
# Simulated code change on 2025-03-10T10:32:01
# Simulated code change on 2025-03-10T16:30:47
# Simulated code change on 2025-03-10T17:24:09
# Simulated code change on 2025-03-10T22:23:40
# Simulated code change on 2025-03-10T17:36:25
# Simulated code change on 2025-03-10T19:08:57
# Simulated code change on 2025-03-11T16:25:34
# Simulated code change on 2025-03-11T15:27:59
# Simulated code change on 2025-03-11T11:13:33
# Simulated code change on 2025-03-11T11:11:39
# Simulated code change on 2025-03-11T17:30:36
# Simulated code change on 2025-03-11T22:10:51
# Simulated code change on 2025-03-11T11:56:58
# Simulated code change on 2025-03-12T20:52:57
# Simulated code change on 2025-03-12T10:09:44
# Simulated code change on 2025-03-12T14:20:05
# Simulated code change on 2025-03-12T23:29:06
# Simulated code change on 2025-03-12T21:10:18
# Simulated code change on 2025-03-12T17:43:26
# Simulated code change on 2025-03-12T17:13:13
# Simulated code change on 2025-03-12T09:20:11
# Simulated code change on 2025-03-12T16:24:02
# Simulated code change on 2025-03-12T23:00:09
# Simulated code change on 2025-03-12T17:40:53
# Simulated code change on 2025-03-12T23:56:16
# Simulated code change on 2025-03-13T13:12:19
