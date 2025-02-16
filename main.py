from replay_process import *

def run():
    ## Get all replays from replays_all.json
    # with open("config.json", "r") as f:
    #     config = json.load(f)
    # AUTHORIZATION_TOKEN = config.get("authorization_token")
    # headers = { "Authorization": AUTHORIZATION_TOKEN }
    # rf = ReplayFetcher(headers).fetch_replays()


    # Process replays into csv
    # output_file = "replays_stats.json"
    # with open(output_file, "r") as file:
    #     replay_data = json.load(file)

    # convert = ReplayConvert(player_name="Silversphere95")
    # convert_df = convert.replays_json_to_dataframe(replay_data)

    # convert_df.to_csv("replays_processed.csv", index=False, na_rep="NaN")
    # print("Converted replays")


    processor = ModelProcess()

    data = processor.load_and_preprocess_data(r'D:\\Coding\\Python\\RLStatistics\\replays_processed.csv', encoding_type="onehot")
    X_train, X_test, y_train, y_test = processor.split_data(data)
    processor.train_random_forest(X_train, y_train)
    fpr, tpr, roc_auc = processor.evaluate_model(X_test, y_test)

    next_game_data = {
        "rank_tier": 18,
        "min_rank": 18,
        "day_of_week_Wednesday": 1,
        "server_region_USW": 1,
        "games_already_played": 2
    }
    win_probability, is_win = processor.predict_next_game(next_game_data, X_train.columns)
    print(f"Win Probability: {win_probability:.2f}")
    print(f"Predicted Outcome: {'Win' if is_win else 'Loss'}")

    #processor.plot_confusion_matrix(X_test, y_test)
    #processor.plot_heatmap(data)
    #processor.show_plots()


if __name__ == "__main__":
    run()