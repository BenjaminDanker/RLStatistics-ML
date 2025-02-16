import time
import requests
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


class ReplayFetcher:
    def __init__(self, headers, short_term_limit=2, long_term_limit=1000, save_interval=10):
        self.headers = headers
        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit
        self.calls_in_last_second = 0
        self.start_of_second = time.time()
        self.calls_in_last_hour = 0
        self.start_of_hour = time.time()
        self.save_interval = save_interval
        self.old_replays_count = 0
        self.new_replays_count = 0

    def fetch_replays(self, replay_links, output_file):
        # Load existing replays from the output file, if it exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    replays = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Existing output file is corrupted or empty. Starting fresh.")
                    replays = {}
        else:
            replays = {}

        for link in replay_links:
            # Extract the replay ID from the link
            replay_id = link.split("/")[-1]

            # Skip replay if already fetched
            if replay_id in replays:
                self.old_replays_count += 1
                print(f"Replay already fetched: {replay_id}, Total skipped: {self.old_replays_count}")
                continue

            while True:
                current_time = time.time()

                # Check long-term limit
                elapsed_hour = current_time - self.start_of_hour
                if elapsed_hour >= 3600:
                    self.calls_in_last_hour = 0
                    self.start_of_hour = current_time

                if self.calls_in_last_hour >= self.long_term_limit:
                    wait_time = 3600 - elapsed_hour
                    print(f"Hourly limit reached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    self.calls_in_last_hour = 0
                    self.start_of_hour = time.time()
                    continue

                # Check short-term limit
                elapsed_second = current_time - self.start_of_second
                if elapsed_second >= 1:
                    self.calls_in_last_second = 0
                    self.start_of_second = current_time

                if self.calls_in_last_second >= self.short_term_limit:
                    wait_time = 1 - elapsed_second
                    print(f"Short-term limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    self.calls_in_last_second = 0
                    self.start_of_second = time.time()
                    continue

                # Make the request
                response = requests.get(link, headers=self.headers)
                self.calls_in_last_second += 1
                self.calls_in_last_hour += 1

                if response.status_code == 200:
                    data = response.json()
                    replays[replay_id] = data  # Add replay to dictionary
                    self.new_replays_count += 1
                    print(f"Total replays fetched so far: {len(replays)}")

                    # Save periodically
                    if self.new_replays_count >= self.save_interval:
                        with open(output_file, "w") as f:
                            json.dump(replays, f, indent=4)
                        print(f"Saved {len(replays)} replays to {output_file}")
                        self.new_replays_count = 0  # Reset counter

                    break  # Exit retry loop on success

                elif response.status_code == 429:
                    print("Rate limit reached. Waiting 1 second before retrying...")
                    time.sleep(1)
                    continue

                else:
                    print(f"Unexpected status code {response.status_code}. Skipping {link}...")
                    break

        # Final save at the end
        if self.new_replays_count > 0:
            with open(output_file, "w") as f:
                json.dump(replays, f, indent=4)
            print(f"Final save: {len(replays)} replays saved to {output_file}")


class ReplayConvert:
    def __init__(self, player_name=""):
        self.player_name = player_name

    def convert_replay(self, replay):
        try:
            # Determine which team the player is on
            player_team = "blue" if any(player["name"] == self.player_name for player in replay["blue"]["players"]) else "orange"

            # Extract relevant team stats
            blue_stats = replay["blue"]["stats"]["core"]
            orange_stats = replay["orange"]["stats"]["core"]

            # Determine if it was a win for the player
            is_win = (
                (player_team == "blue" and blue_stats["goals"] > orange_stats["goals"])
                or (player_team == "orange" and orange_stats["goals"] > blue_stats["goals"])
            )

            # Extract player-specific stats
            player_data = next(
                player for player in replay[player_team]["players"] if player["name"] == self.player_name
            )
            player_stats = player_data.get("stats", None)
            player_camera = player_data.get("camera", None)

            # Extract rank and min/max rank
            rank_tier = player_data.get("rank", {}).get("tier", None)
            min_rank = replay.get("min_rank", {}).get("tier", None)
            max_rank = replay.get("max_rank", {}).get("tier", None)

            # Extract date and time information
            date_str = replay.get("date")  # Example: "2024-12-10T12:27:37-06:00"
            if date_str:
                replay_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))  # Handle ISO 8601 format
                day_of_week = replay_date.strftime("%A")  # Full day name (e.g., "Tuesday")
            else:
                day_of_week = None
                replay_date = None

            # Extract server region
            server_region = replay.get("server", {}).get("region", None)

            # Flatten data into a row
            row = {
                # General replay info
                "team_size": replay.get("team_size", None),
                "rank_tier": rank_tier,
                "min_rank": min_rank,
                "max_rank": max_rank,
                "is_win": int(is_win),
                "day_of_week": day_of_week,
                "date": replay_date,
                "server_region": server_region,

                # Core stats
                "player_goals": player_stats.get("core", {}).get("goals", None) if player_stats else None,
                "player_saves": player_stats.get("core", {}).get("saves", None) if player_stats else None,
                "player_shots": player_stats.get("core", {}).get("shots", None) if player_stats else None,
                "player_assists": player_stats.get("core", {}).get("assists", None) if player_stats else None,
                "player_score": player_stats.get("core", {}).get("score", None) if player_stats else None,

                # Boost stats
                "player_boost_avg": player_stats.get("boost", {}).get("avg_amount", None) if player_stats else None,
                "player_boost_bpm": player_stats.get("boost", {}).get("bpm", None) if player_stats else None,
                "player_boost_bcpm": player_stats.get("boost", {}).get("bcpm", None) if player_stats else None,
                "player_boost_percent_0_25": player_stats.get("boost", {}).get("percent_boost_0_25", None) if player_stats else None,
                "player_boost_percent_75_100": player_stats.get("boost", {}).get("percent_boost_75_100", None) if player_stats else None,

                # Movement stats
                "player_movement_avg_speed": player_stats.get("movement", {}).get("avg_speed", None) if player_stats else None,
                "player_movement_total_distance": player_stats.get("movement", {}).get("total_distance", None) if player_stats else None,
                "player_movement_time_supersonic": player_stats.get("movement", {}).get("time_supersonic_speed", None) if player_stats else None,
                "player_movement_time_slow": player_stats.get("movement", {}).get("time_slow_speed", None) if player_stats else None,

                # Positioning stats
                "player_positioning_avg_distance_to_ball": player_stats.get("positioning", {}).get("avg_distance_to_ball", None) if player_stats else None,
                "player_positioning_time_defensive": player_stats.get("positioning", {}).get("time_defensive_half", None) if player_stats else None,
                "player_positioning_time_offensive": player_stats.get("positioning", {}).get("time_offensive_half", None) if player_stats else None,

                # Demo stats
                "player_demos_inflicted": player_stats.get("demo", {}).get("inflicted", None) if player_stats else None,
                "player_demos_taken": player_stats.get("demo", {}).get("taken", None) if player_stats else None,

                # Camera settings
                "camera_fov": player_camera.get("fov", None) if player_camera else None,
                "camera_height": player_camera.get("height", None) if player_camera else None,
                "camera_pitch": player_camera.get("pitch", None) if player_camera else None,
                "camera_distance": player_camera.get("distance", None) if player_camera else None,
                "camera_stiffness": player_camera.get("stiffness", None) if player_camera else None,
                "camera_swivel_speed": player_camera.get("swivel_speed", None) if player_camera else None,
                "camera_transition_speed": player_camera.get("transition_speed", None) if player_camera else None,
            }
            return row
        except Exception as e:
            print(f"Error processing replay: {e}")
            return None

    def replays_json_to_dataframe(self, replays):
        # Filter replays with "Ranked" in playlist_name
        filtered_replays = {
            replay_id: replay
            for replay_id, replay in replays.items()
            if "Ranked" in replay.get("playlist_name", "")
        }

        # Process replays into rows
        processed_replays = [
            self.convert_replay(replay) for replay_id, replay in filtered_replays.items() if self.convert_replay(replay) is not None
        ]

        # Convert to DataFrame
        df = pd.DataFrame(processed_replays)

        # Ensure the 'date' column is present
        if "date" in df.columns:
            # Convert 'date' to datetime
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Sort by 'date' to ensure chronological order
            df = df.sort_values("date").reset_index(drop=True)

            # Calculate time gaps in hours and seconds
            df["time_gap_seconds"] = df["date"].diff().dt.total_seconds()  # Gap in seconds
            df["time_gap"] = df["time_gap_seconds"] / 3600  # Gap in hours

            # Check for gaps less than 5 seconds
            for i, row in df.iterrows():
                if i > 0 and row["time_gap_seconds"] < 30:
                    print(f"Gap less than 5 seconds detected: {row['time_gap_seconds']} seconds between rows {i-1} and {i}")

            # Initialize games_already_played
            games_played = []
            counter = 0

            for gap in df["time_gap"]:
                if pd.isna(gap) or gap >= 12:  # Reset counter for a new game day
                    counter = 0
                games_played.append(counter)
                counter += 1

            # Add the games_already_played column
            df["games_already_played"] = games_played

        else:
            print("Error: 'date' column is missing.")
            df["games_already_played"] = None  # Default if no date data is available

        # Drop the helper columns
        df.drop(columns=["time_gap", "time_gap_seconds"], inplace=True)

        # Convert specific columns to categorical
        categorical_features = ["team_size", "rank_tier", "day_of_week", "server_region"]
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype("category")

        return df
    

class PreProcess:
    def preprocess(self, data):
        data = data.dropna(subset=["min_rank", "server_region"])

        data.loc[:, "rank_tier"] = data["rank_tier"].fillna((data["min_rank"] + data["max_rank"]) / 2)
        data.loc[:, "rank_tier"] = data["rank_tier"].round().astype(int)
        data = data.drop(columns=["min_rank", "max_rank", "date", "camera_transition_speed", "camera_fov", "camera_pitch", "camera_distance", "camera_height"])

        return data

    def label_encode(self, data):
        # Manual mapping for server_region
        server_region_mapping = {
            "USE": 1,
            "USW" : 2,
            "USC" : 3,
        }

        # Manual mapping for day_of_week
        day_of_week_mapping = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7,
        }

        # Apply the mappings
        data["server_region"] = data["server_region"].map(server_region_mapping)
        data["day_of_week"] = data["day_of_week"].map(day_of_week_mapping)

        return data
    
    def onehot_encode(self, data):
        # One-hot encode server_region
        server_region_one_hot = pd.get_dummies(data["server_region"], prefix="server_region")

        # One-hot encode day_of_week
        day_of_week_one_hot = pd.get_dummies(data["day_of_week"], prefix="day")

        # Concatenate the one-hot encoded columns with the original dataset
        data = pd.concat([data, server_region_one_hot, day_of_week_one_hot], axis=1)

        # Drop the original columns
        data = data.drop(columns=["server_region", "day_of_week"])

        return data
    
    def split_by_teamsize(self, data):
        team_size_2_data = data[data["team_size"] == 2].drop(columns="team_size")
        team_size_3_data = data[data["team_size"] == 3].drop(columns="team_size")

        return team_size_2_data, team_size_3_data
    

class ModelProcess:
    def __init__(self):
        self.model = None

    def load_and_preprocess_data(self, file_path, rank_tier_minimum=18, encoding_type="onehot"):
        data = pd.read_csv(file_path)

        process = PreProcess()
        team_size_2_data, team_size_3_data = process.split_by_teamsize(data)
        data = team_size_3_data
        data = data[data["rank_tier"] >= rank_tier_minimum]
        data = process.preprocess(data)
        if encoding_type == "onehot":
            data = process.onehot_encode(data)
        elif encoding_type == "label":
            data = process.label_encode(data)
        else:
            raise ValueError("Invalid encoding_type. Choose 'onehot' or 'label'.")

        print(f"Data size after preprocessing: {len(data)}")

        return data

    def split_data(self, data, test_size=0.2, random_state=42):
        X = data.drop(columns=["is_win"])
        y = data["is_win"]
      
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_random_forest(self, X_train, y_train, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=3, n_estimators=300, random_state=42):
        self.model = RandomForestClassifier(
            max_depth = max_depth,
            max_features = max_features,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split,
            n_estimators = n_estimators,
            random_state = random_state
        )
        self.model.fit(X_train, y_train)

        print("Random Forest model trained.")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.3f}")

        return fpr, tpr, roc_auc

    def predict_next_game(self, next_game_data, columns):
        next_game_df = pd.DataFrame([next_game_data]).reindex(columns=columns, fill_value=0)
        win_probability = self.model.predict_proba(next_game_df)[:, 1][0]
        is_win = self.model.predict(next_game_df)[0]

        return win_probability, is_win

    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()

    def plot_heatmap(self, data):
        correlation_matrix = data.corr().replace([np.inf, -np.inf], np.nan).fillna(0)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation'})
        plt.title("Heatmap of All Numerical Data")

    def show_plots(self):
        plt.show()