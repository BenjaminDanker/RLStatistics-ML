# RLStatistics-ML

**RLStatistics-ML** is a Python toolkit designed to process Rocket League replay data, extract player and match statistics, and train machine learning models—such as Random Forest classifiers—to predict match outcomes and enhance an in-game plugin’s analytics capabilities.

## Introduction

RLStatistics-ML provides a streamlined pipeline to:
1. Parse and clean raw replay JSON files.
2. Compute aggregated statistics per player and per match.
3. Train and evaluate a Random Forest model for outcome prediction.

## Features

- **Replay Processing**: Load and normalize Rocket League replay data into pandas DataFrames for further analysis.
- **Statistical Extraction**: Compute key metrics (goals, saves, shots, assists) per player and match.
- **Machine Learning Pipeline**: Train a Random Forest classifier end-to-end using scikit-learn’s `RandomForestClassifier` for match outcome prediction.
- **Configurable Workflows**: Toggle data processing and model training directly in `main.py` via commenting/uncommenting.
- **CSV & JSON Outputs**: Export processed data and model predictions in both CSV and JSON formats for integration with external tools.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BenjaminDanker/RLStatistics-ML.git
   cd RLStatistics-ML
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

All operations are controlled in `main.py`. By commenting or uncommenting the corresponding lines, you can switch between replay processing and model training:

```python
if __name__ == "__main__":
    # To process replay files:
    # from replay_process import process_replays
    # process_replays(input_path="replays_all.json", output_path="replays_processed.csv")

    # To train & evaluate the model:
    # from train_model import train_and_evaluate
    # train_and_evaluate(train_csv="replays_processed.csv", output_json="replays_stats.json")
```

1. **Enable replay processing** by uncommenting the `process_replays` lines.
2. **Enable model training** by uncommenting the `train_and_evaluate` lines.
3. **Run**:
   ```bash
   python main.py
   ```

## Dependencies

- **Python 3.8+**
- **pandas**: data manipulation and analysis library
- **NumPy**: numerical computing library
- **scikit-learn**: machine learning toolkit, used here for Random Forests

## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT [License](License).
