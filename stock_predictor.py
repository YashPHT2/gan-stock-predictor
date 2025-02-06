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
