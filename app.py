"""
NFL Game Predictor - Flask Web Application

Simple web interface to display predictions for upcoming NFL games.
"""

from flask import Flask, render_template
from datetime import datetime
import os
import pandas as pd
from nfl_predictor import (
    load_historical_games,
    load_upcoming_games,
    get_next_week_games,
    train_model,
    predict_upcoming_games,
    load_model,
    save_model
)

app = Flask(__name__)

MODEL_FILE = 'nfl_model.pkl'

def get_predictions():
    """Get predictions for upcoming games."""
    # Load or train model
    if os.path.exists(MODEL_FILE):
        print("Loading saved model...")
        model, feature_columns = load_model(MODEL_FILE)
    else:
        print("Training new model...")
        completed_games, date_col = load_historical_games()
        print("completed_games shape:", completed_games.shape)  # (rows, cols)
        model, feature_columns = train_model(completed_games, date_col)
        save_model(model, feature_columns, MODEL_FILE)
        # Keep completed_games and date_col for predictions
        return get_predictions_with_data(model, feature_columns, completed_games, date_col)
    
    # Load historical games for feature calculation
    completed_games, date_col = load_historical_games()
    
    return get_predictions_with_data(model, feature_columns, completed_games, date_col)

def get_predictions_with_data(model, feature_columns, completed_games, date_col):
    """Get predictions using loaded model and data."""
    # Load upcoming games
    upcoming_games, upcoming_date_col = load_upcoming_games()
    
    if upcoming_games.empty:
        return [], "No upcoming games found. The season may have ended."
    
    # Get next week's games
    use_date_col = date_col or upcoming_date_col
    next_week_games = get_next_week_games(upcoming_games, use_date_col)
    
    if next_week_games.empty:
        return [], "No games scheduled for the next week."
    
    # Make predictions
    predictions = predict_upcoming_games(
        model, 
        feature_columns, 
        next_week_games, 
        completed_games, 
        use_date_col
    )
    
    if predictions.empty:
        return [], "Error making predictions. Please try retraining the model."
    
    # Format for display
    formatted_predictions = []
    for _, pred in predictions.iterrows():
        try:
            game_date_str = "TBD"
            if pd.notna(pred.get('game_date')):
                if hasattr(pred['game_date'], 'strftime'):
                    game_date_str = pred['game_date'].strftime('%B %d, %Y %I:%M %p')
                else:
                    game_date_str = str(pred['game_date'])
            
            formatted_predictions.append({
                'away_team': str(pred['away_team']),
                'home_team': str(pred['home_team']),
                'predicted_winner': str(pred['predicted_winner']),
                'home_win_prob': float(pred['home_win_probability']),
                'away_win_prob': float(pred['away_win_probability']),
                'confidence': float(pred['confidence']),
                'game_date': game_date_str
            })
        except Exception as e:
            print(f"Error formatting prediction: {e}")
            continue
    
    return formatted_predictions, None

@app.route('/')
def index():
    """Main page showing predictions."""
    try:
        predictions, error = get_predictions()
        return render_template('index.html', 
                             predictions=predictions, 
                             error=error,
                             last_updated=datetime.now().strftime('%B %d, %Y at %I:%M %p'))
    except Exception as e:
        return render_template('index.html', 
                             predictions=[], 
                             error=f"Error loading predictions: {str(e)}",
                             last_updated=datetime.now().strftime('%B %d, %Y at %I:%M %p'))

@app.route('/retrain')
def retrain():
    """Retrain the model (takes a while)."""
    try:
        print("Retraining model...")
        completed_games, date_col = load_historical_games()
        model, feature_columns = train_model(completed_games, date_col)
        save_model(model, feature_columns, MODEL_FILE)
        return render_template('index.html', 
                             predictions=[], 
                             error="Model retrained successfully! Refresh the page to see new predictions.",
                             last_updated=datetime.now().strftime('%B %d, %Y at %I:%M %p'))
    except Exception as e:
        return render_template('index.html', 
                             predictions=[], 
                             error=f"Error retraining model: {str(e)}",
                             last_updated=datetime.now().strftime('%B %d, %Y at %I:%M %p'))

if __name__ == '__main__':
    print("Starting NFL Predictor web app...")
    print("Visit http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)

