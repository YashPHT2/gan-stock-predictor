# 📈 Stock Price Prediction using GANs and Sentiment Analysis

This project presents an advanced approach to stock price prediction by combining financial time-series data with public sentiment from social media. It leverages a Generative Adversarial Network (GAN) with LSTM and CNN components to forecast future stock prices for Tesla (TSLA).

---

### 🌟 Core Features

-   **Sentiment Analysis**: Uses NLTK's VADER to process raw tweet data and generate daily sentiment scores.
-   **Technical Indicators**: Enriches the financial data with common technical indicators like Moving Averages (MA), MACD, and Bollinger Bands.
-   **Hybrid GAN Architecture**:
    -   **Generator**: A multi-layered LSTM network designed to learn and generate realistic stock price sequences.
    -   **Discriminator**: A CNN-based network trained to distinguish between real and generator-created price data, pushing the generator to improve.
-   **End-to-End Pipeline**: Includes complete data preprocessing, normalization, model training, and evaluation steps.
-   **Data Visualization**: Generates plots for technical indicators, training loss, and comparison between real and predicted stock prices.

### 🛠️ Technology Stack

-   **Python 3**
-   **TensorFlow / Keras**: For building and training the GAN models.
-   **Pandas & NumPy**: For data manipulation and numerical operations.
-   **Scikit-learn**: For data scaling and metrics.
-   **NLTK (Vader)**: For sentiment analysis of tweet data.
-   **Matplotlib**: For data visualization.
-   **Statsmodels**: For statistical computations.

### 🚀 Getting Started

#### Prerequisites

-   Python 3.8 or higher
-   The datasets used in this project (`stock_tweets.csv`, `stock_yfinance_data.csv`). Place them in a `/data` directory.

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run a Python interpreter and execute the following:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

#### Running the Script

Execute the main script from your terminal:

```bash
python stock_predictor.py
```
The script will process the data, train the models, and generate output plots.

---

### ⚙️ Project Structure

```
.
├── stock_predictor.py   # Main script with all logic
├── requirements.txt     # Project dependencies
├── README.md            # You are here
└── .gitignore           # Specifies files for Git to ignore
