"""
NFL Game Predictor - Core prediction functions

This module contains functions for loading data, training models, and making predictions.
"""

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
    NFL_DATA_PY_AVAILABLE = True
except ImportError:
    NFL_DATA_PY_AVAILABLE = False

def get_team_stats_from_history(team, games_df, date_col, current_date, window_size=16):
    """Calculate team statistics from historical games up to a given date."""
    if date_col and date_col in games_df.columns:
        team_games = games_df[
            (games_df[date_col] < current_date) & 
            ((games_df['home_team'] == team) | (games_df['away_team'] == team))
        ].tail(window_size)
    else:
        # If no date column, just get all games for the team
        team_games = games_df[
            (games_df['home_team'] == team) | (games_df['away_team'] == team)
        ].tail(window_size)
    
    points_scored = []
    points_allowed = []
    wins = 0
    games_count = 0
    
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            points_scored.append(game['home_score'])
            points_allowed.append(game['away_score'])
            wins += game['home_win']
            games_count += 1
        elif game['away_team'] == team:
            points_scored.append(game['away_score'])
            points_allowed.append(game['home_score'])
            wins += (1 - game['home_win'])
            games_count += 1
    
    avg_scored = np.mean(points_scored) if points_scored else 21.0
    avg_allowed = np.mean(points_allowed) if points_allowed else 21.0
    win_rate = wins / games_count if games_count > 0 else 0.5
    point_diff = avg_scored - avg_allowed
    
    return {
        'avg_points_scored': avg_scored,
        'avg_points_allowed': avg_allowed,
        'win_rate': win_rate,
        'point_differential': point_diff
    }

def calculate_features_for_game(game, completed_games_df, date_col, window_size=16):
    """Calculate features for a single game based on historical data."""
    home_team = game.get('home_team')
    away_team = game.get('away_team')
    
    # Get game date if available
    if date_col and date_col in game:
        game_date = pd.to_datetime(game.get(date_col, datetime.now()))
    else:
        game_date = datetime.now()
    
    # Get stats for both teams
    home_stats = get_team_stats_from_history(home_team, completed_games_df, date_col, game_date, window_size)
    away_stats = get_team_stats_from_history(away_team, completed_games_df, date_col, game_date, window_size)
    
    # Create feature dictionary
    features = {
        'home_avg_points_scored': home_stats['avg_points_scored'],
        'away_avg_points_scored': away_stats['avg_points_scored'],
        'home_avg_points_allowed': home_stats['avg_points_allowed'],
        'away_avg_points_allowed': away_stats['avg_points_allowed'],
        'home_win_rate': home_stats['win_rate'],
        'away_win_rate': away_stats['win_rate'],
        'home_point_differential': home_stats['point_differential'],
        'away_point_differential': away_stats['point_differential'],
    }
    
    # Add difference features
    features['points_scored_diff'] = features['home_avg_points_scored'] - features['away_avg_points_scored']
    features['points_allowed_diff'] = features['home_avg_points_allowed'] - features['away_avg_points_allowed']
    features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
    features['point_differential_diff'] = features['home_point_differential'] - features['away_point_differential']
    
    return features

def load_historical_games(seasons=None):
    """Load historical NFL games from nfl_data_py.
    Includes both regular season and postseason/playoff games."""
    if not NFL_DATA_PY_AVAILABLE:
        raise ImportError("nfl_data_py is required. Install with: pip install nfl-data-py")
    
    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(2020, current_year + 1))
    
    all_games = []
    for season in seasons:
        try:
            # nfl_data_py schedule includes regular season and postseason games
            season_games = nfl.import_schedules([season])
            if len(season_games) > 0:
                all_games.append(season_games)
        except:
            pass
    
    if not all_games:
        raise ValueError("Could not load any game data")
    
    games = pd.concat(all_games, ignore_index=True)
    
    # Filter to completed games only (includes regular season and playoff games)
    games = games[games['home_score'].notna() & games['away_score'].notna()].copy()
    
    # Sort by date
    date_col = None
    if 'gameday' in games.columns:
        date_col = 'gameday'
        games[date_col] = pd.to_datetime(games[date_col], errors='coerce')
    elif 'game_date' in games.columns:
        date_col = 'game_date'
        games[date_col] = pd.to_datetime(games[date_col], errors='coerce')
    
    if date_col:
        games = games.sort_values(date_col).reset_index(drop=True)
    
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    
    return games, date_col

def load_upcoming_games(season=None):
    """Load NFL games for predictions - includes current week games even if they've started.
    Includes both regular season and postseason/playoff games.
    Note: Playoff games for a season occur in January/February of the next year."""
    if not NFL_DATA_PY_AVAILABLE:
        raise ImportError("nfl_data_py is required. Install with: pip install nfl-data-py")
    
    if season is None:
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        # If we're in January or February, we might be in postseason
        # Playoff games belong to the previous year's season (e.g., Jan 2025 = 2024 season playoffs)
        if current_month <= 2:
            # Check both current year and previous year for playoff games
            seasons_to_load = [current_year - 1, current_year]
        else:
            # Regular season, use current year
            seasons_to_load = [current_year]
    else:
        seasons_to_load = [season]
    
    try:
        all_games_list = []
        for season_to_load in seasons_to_load:
            try:
                season_games = nfl.import_schedules([season_to_load])
                if len(season_games) > 0:
                    all_games_list.append(season_games)
            except Exception as e:
                print(f"Warning: Could not load season {season_to_load}: {e}")
                continue
        
        if not all_games_list:
            return pd.DataFrame(), None
        
        all_games = pd.concat(all_games_list, ignore_index=True)
        
        # Get date column
        date_col = None
        if 'gameday' in all_games.columns:
            date_col = 'gameday'
            all_games[date_col] = pd.to_datetime(all_games[date_col], errors='coerce')
        elif 'game_date' in all_games.columns:
            date_col = 'game_date'
            all_games[date_col] = pd.to_datetime(all_games[date_col], errors='coerce')
        
        if date_col:
            all_games = all_games.sort_values(date_col).reset_index(drop=True)
        
        # Return all games (we'll filter by date in get_next_week_games)
        # This includes both regular season and postseason games
        return all_games, date_col
    except Exception as e:
        print(f"Error loading games: {e}")
        return pd.DataFrame(), None

def get_next_week_games(upcoming_games_df, date_col):
    """Get games for the current week. 
    Shows games even after they start, until Tuesday night when we update for next week.
    On Tuesday night, filters out completed games from previous week."""
    if upcoming_games_df.empty:
        return pd.DataFrame()
    
    if date_col and date_col in upcoming_games_df.columns:
        now = datetime.now()
        today = now.date()
        current_weekday = now.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
        
        # Convert date column to date
        upcoming_games_df = upcoming_games_df.copy()
        upcoming_games_df['game_date_only'] = pd.to_datetime(upcoming_games_df[date_col], errors='coerce').dt.date
        
        # NFL week typically runs Tuesday (after MNF) to Monday (MNF)
        # Determine the start of current week based on day
        # On Tuesday 8pm+, we transition to the new week
        # Otherwise, we're showing the current/ongoing week
        
        if current_weekday == 1 and now.hour >= 20:  # Tuesday 8pm or later
            # New week starts - show games from this Tuesday through next Monday
            week_start = today  # Today (Tuesday)
            week_end = today + timedelta(days=6)  # Next Monday
        elif current_weekday == 1:  # Tuesday before 8pm
            # Still showing previous week's games (last Tuesday through last Monday)
            week_start = today - timedelta(days=7)  # Last Tuesday
            week_end = today - timedelta(days=1)  # Last Monday
        elif current_weekday == 0:  # Monday
            # Show current week (last Tuesday to this Monday)
            week_start = today - timedelta(days=6)  # Last Tuesday
            week_end = today  # Today (Monday)
        else:  # Wednesday-Sunday
            # Show current week (last Tuesday to next Monday)
            days_since_tuesday = (current_weekday - 1) if current_weekday > 1 else 0
            week_start = today - timedelta(days=days_since_tuesday)
            week_end = today + timedelta(days=(8 - current_weekday))  # Next Monday
        
        # Filter games in the current week (including games that have started)
        current_week_games = upcoming_games_df[
            (upcoming_games_df['game_date_only'] >= week_start) & 
            (upcoming_games_df['game_date_only'] <= week_end)
        ].copy()
        
        # On Tuesday night (8pm+), we've already filtered to the new week
        # so no need to filter out old games - they're already excluded
        
        if not current_week_games.empty:
            return current_week_games
    
    # Fallback: if no date column or no games in current week, return upcoming games
    # Include games from recent past to near future
    if date_col and date_col in upcoming_games_df.columns:
        today = datetime.now().date()
        recent_games = upcoming_games_df[
            (upcoming_games_df['game_date_only'] >= today - timedelta(days=7)) &
            (upcoming_games_df['game_date_only'] <= today + timedelta(days=7))
        ].copy()
        if not recent_games.empty:
            return recent_games
    
    # Last resort: return first 16 games
    return upcoming_games_df.head(16).copy()

def train_model(completed_games_df, date_col, window_size=16):
    """Train a logistic regression model on historical games."""
    print("Calculating features for historical games...")
    
    game_features_list = []
    games_sorted = completed_games_df.sort_values(date_col if date_col else 'game_id').reset_index(drop=True)
    
    for idx, game in games_sorted.iterrows():
        game_date = pd.to_datetime(game.get(date_col, datetime.now()))
        features = calculate_features_for_game(game, games_sorted, date_col, window_size)
        
        game_features_list.append({
            'game_id': game.get('game_id', idx),
            'season': game.get('season'),
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_win': game['home_win'],
            **features
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(games_sorted)} games...")
    
    game_features = pd.DataFrame(game_features_list)
    
    # Define feature columns
    feature_columns = [
        'home_avg_points_scored', 'away_avg_points_scored',
        'home_avg_points_allowed', 'away_avg_points_allowed',
        'home_win_rate', 'away_win_rate',
        'home_point_differential', 'away_point_differential',
        'points_scored_diff', 'points_allowed_diff',
        'win_rate_diff', 'point_differential_diff'
    ]
    
    X = game_features[feature_columns].fillna(0)
    y = game_features['home_win']
    
    # Filter invalid rows
    valid_mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nTraining model on {len(X)} games...")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    train_accuracy = model.score(X, y)
    print(f"Model accuracy: {train_accuracy:.3f}")
    
    return model, feature_columns

def predict_upcoming_games(model, feature_columns, upcoming_games_df, completed_games_df, date_col, window_size=16):
    """Make predictions for upcoming games."""
    predictions = []
    
    for _, game in upcoming_games_df.iterrows():
        try:
            features_dict = calculate_features_for_game(game, completed_games_df, date_col, window_size)
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features_dict])
            features_df = features_df[feature_columns].fillna(0)
            
            # Make prediction
            home_win_prob = model.predict_proba(features_df)[0][1]
            predicted_winner = game['home_team'] if home_win_prob > 0.5 else game['away_team']
            
            predictions.append({
                'game_id': game.get('game_id', ''),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game.get(date_col, ''),
                'predicted_winner': predicted_winner,
                'home_win_probability': home_win_prob,
                'away_win_probability': 1 - home_win_prob,
                'confidence': abs(home_win_prob - 0.5) * 2  # 0 to 1 scale
            })
        except Exception as e:
            print(f"Error predicting game {game.get('game_id', 'unknown')}: {e}")
    
    return pd.DataFrame(predictions)

def save_model(model, feature_columns, filepath='nfl_model.pkl'):
    """Save trained model and feature columns."""
    model_data = {
        'model': model,
        'feature_columns': feature_columns
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath='nfl_model.pkl'):
    """Load trained model and feature columns."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_columns']

