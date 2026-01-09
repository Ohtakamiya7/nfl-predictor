# Testing Guide for NFL Predictor

## Quick Start

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn flask nfl-data-py
```

### 2. Test the Functionality

Run the test script to verify everything works:

```bash
python test_predictions.py
```

This will:
- Load historical games (including playoffs)
- Load upcoming games (including games that have started)
- Test the current week game filtering
- Train or load the model
- Generate sample predictions

### 3. Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The app will start on **http://localhost:5001**

Open your browser and visit: **http://localhost:5001**

### 4. What to Test

#### Test 1: Predictions Persist After Games Start
1. Note which games are shown on the homepage
2. Wait for games to start (or check on a day when games are happening)
3. Refresh the page - the games should still be visible
4. Verify predictions are still shown for games in progress

#### Test 2: Tuesday Night Update
1. Check the page on Tuesday night (after 8pm)
2. Games from the previous week should be replaced with next week's games
3. You can simulate this by temporarily changing the system date/time

#### Test 3: Postseason Games
1. Check during playoff season (January)
2. Verify playoff games appear in the predictions
3. The schedule should include Wild Card, Divisional, Conference, and Super Bowl games

#### Test 4: Retrain Model
1. Visit **http://localhost:5001/retrain** to retrain the model with latest data
2. This may take a minute or two
3. Refresh the homepage to see updated predictions

## Manual Testing Commands

You can also test individual components in a Python shell:

```python
from nfl_predictor import *
from datetime import datetime

# Load games
completed_games, date_col = load_historical_games()
print(f"Loaded {len(completed_games)} completed games")

all_games, date_col = load_upcoming_games()
print(f"Loaded {len(all_games)} total games")

# Get current week games
next_week = get_next_week_games(all_games, date_col)
print(f"Current week has {len(next_week)} games")

# Check if games that have started are included
started = next_week[
    (next_week['home_score'].notna()) | 
    (next_week['away_score'].notna())
]
print(f"Games that have started: {len(started)}")
```

## Troubleshooting

### Issue: No games showing
- Check if it's the off-season (no games scheduled)
- Try retraining the model: visit `/retrain`
- Check internet connection (nfl-data-py requires internet)

### Issue: Model training fails
- Ensure you have enough memory
- Check that historical games are loading correctly
- Try deleting `nfl_model.pkl` and retraining

### Issue: Games disappear too early
- The current logic shows games until Tuesday night (8pm)
- Games from the current week should persist even after they start
- Check the current day/time to understand the filtering logic

### Issue: Postseason games not showing
- Verify you're in the postseason period (January-February)
- Check that the current season year is correct
- The schedule should automatically include playoff games

## Expected Behavior

✅ **Correct**: Games shown even after they start (during current week)
✅ **Correct**: Games update on Tuesday night (8pm+) for new week
✅ **Correct**: Postseason/playoff games included in schedule
✅ **Correct**: Predictions based on historical data up to game date

❌ **Incorrect**: Games disappear as soon as they start
❌ **Incorrect**: Only regular season games shown (no playoffs)
❌ **Incorrect**: Predictions change based on in-game scores

