"""
NFL Game Predictor using Logistic Regression

This script loads NFL game data, calculates team statistics using rolling windows,
and trains a logistic regression model to predict game outcomes.

Data Sources:
- Primary: nfl_data_py (sources from nflfastR, updates nightly during season)
- Alternative: sportsdataverse.nfl package
  - Install: pip install sportsdataverse
  - Usage: from sportsdataverse.nfl import nfl_load_pbp

Note: nfl_data_py updates with current season data, but may be incomplete early in the season.
The script automatically detects available seasons and handles missing data.
"""

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
    NFL_DATA_PY_AVAILABLE = True
except ImportError:
    NFL_DATA_PY_AVAILABLE = False
    print("Warning: nfl_data_py not available. Please install with: pip install nfl-data-py")
    print("\nAlternative: You can use sportsdataverse.nfl instead:")
    print("  pip install sportsdataverse")
    print("  from sportsdataverse.nfl import nfl_load_pbp")

# ------ Prepare Data ------

print("="*60)
print("Loading NFL data...")
print("="*60)

# Determine current year and available seasons
current_year = datetime.now().year
# Try to load recent seasons (2020-2025)
# We'll check which ones actually have data
possible_seasons = list(range(2020, current_year + 2))  # Include next year in case we're mid-season

if NFL_DATA_PY_AVAILABLE:
    # Try to load schedules and see which seasons have data
    print(f"\nAttempting to load data from seasons: {possible_seasons}")
    
    all_games = []
    available_seasons = []
    
    for season in possible_seasons:
        try:
            season_games = nfl.import_schedules([season])
            if len(season_games) > 0:
                # Check if any games have scores (completed games)
                completed = season_games[(season_games['home_score'].notna()) & 
                                        (season_games['away_score'].notna())]
                if len(completed) > 0:
                    all_games.append(season_games)
                    available_seasons.append(season)
                    print(f"  ✓ Season {season}: {len(completed)} completed games")
                else:
                    print(f"  ⚠ Season {season}: Schedule found but no completed games yet")
            else:
                print(f"  ✗ Season {season}: No data available")
        except Exception as e:
            print(f"  ✗ Season {season}: Error loading - {str(e)[:50]}")
    
    if all_games:
        games = pd.concat(all_games, ignore_index=True)
        print(f"\n✓ Successfully loaded data from {len(available_seasons)} seasons: {available_seasons}")
    else:
        raise ValueError("Could not load any game data. Please check your internet connection and nfl_data_py installation.")
    
    # Filter to only completed games (games that have scores)
    games = games[games['home_score'].notna() & games['away_score'].notna()].copy()
    
    # Sort by date to calculate rolling statistics correctly
    if 'gameday' in games.columns:
        games['gameday'] = pd.to_datetime(games['gameday'], errors='coerce')
        games = games.sort_values('gameday').reset_index(drop=True)
    elif 'game_date' in games.columns:
        games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce')
        games = games.sort_values('game_date').reset_index(drop=True)
    
    print(f"\nTotal completed games loaded: {len(games)}")
    print(f"Seasons in dataset: {sorted(games['season'].unique())}")
    
    # Check current season data availability
    if current_year in games['season'].unique():
        current_season_games = games[games['season'] == current_year]
        print(f"\nCurrent season ({current_year}): {len(current_season_games)} completed games")
        if len(current_season_games) < 10:
            print(f"  ⚠ Warning: Very few games for current season. Data may be incomplete.")
    else:
        print(f"\n⚠ Warning: No data found for current season ({current_year})")
        print(f"  Available seasons: {sorted(games['season'].unique())}")
    
else:
    raise ImportError("nfl_data_py is required. Install with: pip install nfl-data-py")

# Create target variable: 1 if home team wins, 0 if away team wins
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# ------ Calculate Team Statistics with Rolling Windows ------

print("\n" + "="*60)
print("Calculating team statistics using rolling windows...")
print("="*60)

# Create a function to calculate rolling statistics up to each game
def calculate_rolling_stats(games_df, window_size=16):
    """
    Calculate rolling statistics for each team up to (but not including) each game.
    This prevents data leakage - we only use games that happened before the current game.
    """
    game_features_list = []
    
    # Sort by date to ensure chronological order
    date_col = None
    if 'gameday' in games_df.columns:
        date_col = 'gameday'
    elif 'game_date' in games_df.columns:
        date_col = 'game_date'
    
    games_sorted = games_df.sort_values(date_col if date_col else 'game_id').reset_index(drop=True)
    
    print(f"  Processing {len(games_sorted)} games with window size of {window_size} games...")
    
    for idx, game in games_sorted.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = game.get(date_col, pd.Timestamp('2020-01-01')) if date_col else None
        
        # Get all games before this one for home team
        if date_col:
            home_prev_games = games_sorted[
                (games_sorted[date_col] < game_date) & 
                ((games_sorted['home_team'] == home_team) | (games_sorted['away_team'] == home_team))
            ].tail(window_size)
        else:
            home_prev_games = games_sorted.iloc[:idx][
                (games_sorted.iloc[:idx]['home_team'] == home_team) | 
                (games_sorted.iloc[:idx]['away_team'] == home_team)
            ].tail(window_size)
        
        # Get all games before this one for away team
        if date_col:
            away_prev_games = games_sorted[
                (games_sorted[date_col] < game_date) & 
                ((games_sorted['home_team'] == away_team) | (games_sorted['away_team'] == away_team))
            ].tail(window_size)
        else:
            away_prev_games = games_sorted.iloc[:idx][
                (games_sorted.iloc[:idx]['home_team'] == away_team) | 
                (games_sorted.iloc[:idx]['away_team'] == away_team)
            ].tail(window_size)
        
        # Calculate home team stats
        home_points_scored = []
        home_points_allowed = []
        home_wins = 0
        home_games_count = 0
        
        for _, prev_game in home_prev_games.iterrows():
            if prev_game['home_team'] == home_team:
                home_points_scored.append(prev_game['home_score'])
                home_points_allowed.append(prev_game['away_score'])
                home_wins += prev_game['home_win']
                home_games_count += 1
            elif prev_game['away_team'] == home_team:
                home_points_scored.append(prev_game['away_score'])
                home_points_allowed.append(prev_game['home_score'])
                home_wins += (1 - prev_game['home_win'])  # Away team win
                home_games_count += 1
        
        # Calculate away team stats
        away_points_scored = []
        away_points_allowed = []
        away_wins = 0
        away_games_count = 0
        
        for _, prev_game in away_prev_games.iterrows():
            if prev_game['home_team'] == away_team:
                away_points_scored.append(prev_game['home_score'])
                away_points_allowed.append(prev_game['away_score'])
                away_wins += prev_game['home_win']
                away_games_count += 1
            elif prev_game['away_team'] == away_team:
                away_points_scored.append(prev_game['away_score'])
                away_points_allowed.append(prev_game['home_score'])
                away_wins += (1 - prev_game['home_win'])  # Away team win
                away_games_count += 1
        
        # Calculate averages (use all-time average if not enough games)
        home_avg_scored = np.mean(home_points_scored) if home_points_scored else 21.0
        home_avg_allowed = np.mean(home_points_allowed) if home_points_allowed else 21.0
        home_win_rate = home_wins / home_games_count if home_games_count > 0 else 0.5
        
        away_avg_scored = np.mean(away_points_scored) if away_points_scored else 21.0
        away_avg_allowed = np.mean(away_points_allowed) if away_points_allowed else 21.0
        away_win_rate = away_wins / away_games_count if away_games_count > 0 else 0.5
        
        # Calculate point differentials
        home_point_diff = home_avg_scored - home_avg_allowed
        away_point_diff = away_avg_scored - away_avg_allowed
        
        # Store features for this game
        game_features_list.append({
            'game_id': game.get('game_id', idx),
            'season': game.get('season', None),
            'home_team': home_team,
            'away_team': away_team,
            'home_win': game['home_win'],
            'home_avg_points_scored': home_avg_scored,
            'home_avg_points_allowed': home_avg_allowed,
            'home_win_rate': home_win_rate,
            'home_point_differential': home_point_diff,
            'away_avg_points_scored': away_avg_scored,
            'away_avg_points_allowed': away_avg_allowed,
            'away_win_rate': away_win_rate,
            'away_point_differential': away_point_diff,
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(games_sorted)} games...")
    
    return pd.DataFrame(game_features_list)

# Calculate rolling statistics (use last 16 games as default - roughly one season)
print("\nNote: Using rolling window statistics (last 16 games) for better predictions")
print("      This prevents data leakage and uses recent team performance\n")

game_features = calculate_rolling_stats(games, window_size=16)

# Create difference features (home - away)
game_features['points_scored_diff'] = game_features['home_avg_points_scored'] - game_features['away_avg_points_scored']
game_features['points_allowed_diff'] = game_features['home_avg_points_allowed'] - game_features['away_avg_points_allowed']
game_features['win_rate_diff'] = game_features['home_win_rate'] - game_features['away_win_rate']
game_features['point_differential_diff'] = game_features['home_point_differential'] - game_features['away_point_differential']

print(f"\n✓ Calculated features for {len(game_features)} games")
print(f"\nSample of calculated features:")
print(game_features[['home_team', 'away_team', 'home_win', 'home_avg_points_scored', 
                     'away_avg_points_scored', 'win_rate_diff', 'point_differential_diff']].head(10))

# Fill any NaN values with 0
game_features = game_features.fillna(0)

# ------ Prepare Data for Logistic Regression ------

# Select feature columns for the model
feature_columns = [
    'home_avg_points_scored', 'away_avg_points_scored',
    'home_avg_points_allowed', 'away_avg_points_allowed',
    'home_win_rate', 'away_win_rate',
    'home_point_differential', 'away_point_differential',
    'points_scored_diff', 'points_allowed_diff',
    'win_rate_diff', 'point_differential_diff'
]

# Remove rows with any missing features
X = game_features[feature_columns].copy()
y = game_features['home_win'].copy()

# Filter out rows with NaN or infinite values
valid_mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
X = X[valid_mask]
y = y[valid_mask]
game_features_clean = game_features[valid_mask].copy()

print(f"\nFinal dataset: {len(X)} games with {len(feature_columns)} features")
print(f"Home win rate in dataset: {y.mean():.3f}")
print(f"\nFeature summary statistics:")
print(X.describe())

# ------ Ready for Logistic Regression ------

print("\n" + "="*60)
print("Data is now prepared and ready for logistic regression!")
print("="*60)
print("\nYou can now train a model using:")
print("  model = LogisticRegression()")
print("  model.fit(X, y)")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature names: {feature_columns}")

# ------ Train Logistic Regression Model ------

print("\n" + "="*60)
print("Training Logistic Regression Model...")
print("="*60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} games")
print(f"Test set: {len(X_test)} games")

# Train the model
print("\nTraining model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\n✓ Model trained successfully!")
print(f"  Training Accuracy: {train_accuracy:.3f}")
print(f"  Test Accuracy: {test_accuracy:.3f}")

# Show feature importance (coefficients)
print(f"\nTop 5 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance.head())


