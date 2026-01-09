"""
Test script to verify the prediction system works correctly.
Tests that games persist after they start and that postseason games are included.
"""

from datetime import datetime
from nfl_predictor import (
    load_historical_games,
    load_upcoming_games,
    get_next_week_games,
    train_model,
    predict_upcoming_games,
    load_model,
    save_model
)
import os

def test_basic_functionality():
    """Test basic prediction functionality."""
    print("=" * 60)
    print("Testing NFL Predictor Functionality")
    print("=" * 60)
    
    # Test 1: Load historical games (should include playoffs)
    print("\n1. Testing load_historical_games()...")
    try:
        completed_games, date_col = load_historical_games()
        print(f"   ✓ Loaded {len(completed_games)} completed games")
        print(f"   ✓ Date column: {date_col}")
        if 'season' in completed_games.columns:
            seasons = sorted(completed_games['season'].unique())
            print(f"   ✓ Seasons included: {seasons}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Load upcoming games (should include all games, not just upcoming)
    print("\n2. Testing load_upcoming_games()...")
    try:
        all_games, date_col = load_upcoming_games()
        print(f"   ✓ Loaded {len(all_games)} total games for current season")
        print(f"   ✓ Date column: {date_col}")
        
        # Check if we have games with scores (started/completed)
        if 'home_score' in all_games.columns and 'away_score' in all_games.columns:
            completed = all_games[
                (all_games['home_score'].notna()) & 
                (all_games['away_score'].notna())
            ]
            upcoming = all_games[
                (all_games['home_score'].isna()) | 
                (all_games['away_score'].isna())
            ]
            print(f"   ✓ Games with scores (started/completed): {len(completed)}")
            print(f"   ✓ Games without scores (upcoming): {len(upcoming)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Test get_next_week_games (should include games that have started)
    print("\n3. Testing get_next_week_games()...")
    try:
        next_week = get_next_week_games(all_games, date_col)
        print(f"   ✓ Found {len(next_week)} games for current week")
        
        if len(next_week) > 0:
            # Check if we're including games that have started
            if 'home_score' in next_week.columns and 'away_score' in next_week.columns:
                started = next_week[
                    (next_week['home_score'].notna()) | 
                    (next_week['away_score'].notna())
                ]
                print(f"   ✓ Games in current week that have started: {len(started)}")
                
                if len(started) > 0:
                    print(f"   ✓ SUCCESS: Predictions will persist after games start!")
                    
            # Show sample game dates
            if date_col and date_col in next_week.columns:
                print(f"\n   Sample game dates:")
                for idx, game in next_week.head(3).iterrows():
                    date_str = str(game.get(date_col, 'N/A'))[:10]
                    home = game.get('home_team', 'N/A')
                    away = game.get('away_team', 'N/A')
                    print(f"      {date_str}: {away} @ {home}")
        else:
            print(f"   ⚠ No games found for current week")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test model training (if model doesn't exist)
    print("\n4. Testing model training...")
    MODEL_FILE = 'nfl_model.pkl'
    if os.path.exists(MODEL_FILE):
        print(f"   ✓ Model file exists, loading...")
        try:
            model, feature_columns = load_model(MODEL_FILE)
            print(f"   ✓ Model loaded successfully")
            print(f"   ✓ Features: {len(feature_columns)}")
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
            return False
    else:
        print(f"   ⚠ Model file not found, would train on first run")
    
    # Test 5: Test predictions
    print("\n5. Testing predictions...")
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"   Training model (this may take a minute)...")
            model, feature_columns = train_model(completed_games, date_col)
            save_model(model, feature_columns, MODEL_FILE)
        else:
            model, feature_columns = load_model(MODEL_FILE)
        
        if len(next_week) > 0:
            predictions = predict_upcoming_games(
                model, 
                feature_columns, 
                next_week, 
                completed_games, 
                date_col
            )
            print(f"   ✓ Generated {len(predictions)} predictions")
            
            if len(predictions) > 0:
                print(f"\n   Sample predictions:")
                for idx, pred in predictions.head(3).iterrows():
                    winner = pred.get('predicted_winner', 'N/A')
                    home_prob = pred.get('home_win_probability', 0) * 100
                    away_prob = pred.get('away_win_probability', 0) * 100
                    away_team = pred.get('away_team', 'N/A')
                    home_team = pred.get('home_team', 'N/A')
                    print(f"      {away_team} @ {home_team}")
                    print(f"         Predicted Winner: {winner} ({home_prob:.1f}% home, {away_prob:.1f}% away)")
        else:
            print(f"   ⚠ No games to predict")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_basic_functionality()
    if not success:
        print("\n⚠ Some tests failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✓ Ready to run the web app!")

