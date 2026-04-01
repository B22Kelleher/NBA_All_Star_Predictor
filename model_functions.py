# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ipywidgets import interact, Dropdown
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ndcg_score




def cross_val_splits(df, years_per_split = 5, train_window = 20):
    """Create 5-year splits where each split trains on all years except a 5-year validation window"""
    seasons = sorted(df['season'].unique())
    splits = []
    
    # Create 5-year validation splits, train on all other years
    for i in range(0, len(seasons) - years_per_split + 1, years_per_split):
        val_years = seasons[i:i + years_per_split]

        val_start = val_years[0]
        train_start = val_start - train_window
        train_end = val_start - 1

        train_years = [year for year in seasons 
                       if train_start <= year <= train_end]

        # Skip fold if not enough training data
        if len(train_years) < train_window:
            continue

        train_idx = df[df['season'].isin(train_years)].index
        val_idx = df[df['season'].isin(val_years)].index

        splits.append((train_idx, val_idx, train_years, val_years))

    return splits


def run_cross_val(model, data, features, num_of_selections = 12, target='next_yr_all_star'):
    results = []
    splits = cross_val_splits(data)

    for fold, (train_idx, val_idx, train_years, val_years) in enumerate(splits, start=1):

        train_df = data.loc[train_idx]
        val_df = data.loc[val_idx]
        train_group = train_df.groupby('season').size().values
        model.fit(train_df[features], train_df[target],group=train_group)
        val_score = model.predict(val_df[features])

        val_df_copy = val_df.copy()
        val_df_copy['val_scores'] = val_score
        val_df_copy['predicted_all_star'] = 0
        for season in val_df_copy['season'].unique():
            season_mask = val_df_copy['season'] == season
            season_data = val_df_copy[season_mask].copy()

            season_data_sorted = season_data.sort_values('val_scores', ascending=False)

            top_n_indices = season_data_sorted.head(num_of_selections).index

            val_df_copy.loc[top_n_indices, 'predicted_all_star'] = 1

        data.loc[val_df_copy.index, 'val_scores'] = val_df_copy['val_scores']
        data.loc[val_df_copy.index, 'predicted_all_star'] = val_df_copy['predicted_all_star']

    
        season_metrics = []
        for season, group in val_df_copy.groupby('season'):
            y_true = group[target].values
            y_pred = group['predicted_all_star'].values
            y_prob = group['val_scores'].values

            season_metrics.append({
                'season': season,
                'precision@12': precision_score(y_true, y_pred, zero_division=0),
                'recall@12': recall_score(y_true, y_pred, zero_division=0),
                'f1@12': f1_score(y_true, y_pred, zero_division=0),
                'ndcg@12': ndcg_score([y_true], [y_prob], k=12)
            })
        season_df = pd.DataFrame(season_metrics)
        metrics = {
            'fold': fold,
            'train_seasons': sorted(train_years),
            'val_seasons': sorted(val_years),
            'precision@12': season_df['precision@12'].mean(),
            'recall@12': season_df['recall@12'].mean(),
            'f1@12': season_df['f1@12'].mean(),
            'ndcg@12': season_df['ndcg@12'].mean()
        }

        results.append(metrics)

    return pd.DataFrame(results), data


def select_primary_team(team_series,players):
    """
    Selects the player's main team for a season.
    Rules:
    1. Ignore '2TM' rows (combined stat rows).
    2. If one team has more than double the games of any other, use that team.
    3. Otherwise, pick the first non-'2TM' entry.
    """
    group = players.loc[team_series.index]
    team_games = group.groupby('team')['g'].sum().drop(index='2TM', errors='ignore')
    top_team = team_games.idxmax()
    if len(team_games) > 1:
        second_most = team_games.sort_values(ascending=False).iloc[1]
        if team_games[top_team] > 2 * second_most:
            return top_team
    non_2tm = [t for t in team_series if t != '2TM']
    return non_2tm[0] if non_2tm else team_series.iloc[0]

def fill_missing_seasons(players_df):
    """Ensure each player has continuous seasons from first to last, filling missing ones with blank entries."""
    all_rows = []
    for pid, group in players_df.groupby('player_id', group_keys=False):
        min_season, max_season = 2000, group['season'].max()
        all_seasons = set(range(min_season, max_season + 1))
        existing = set(group['season'])
        missing = all_seasons - existing

        all_rows.append(group)

        for season in missing:
            prior_seasons = group[group['season'] < season]
            if not prior_seasons.empty:
                last_team = prior_seasons.iloc[-1]['team']
            else:
                last_team = np.nan
       
        for season in missing:
            blank_row = {
                'player_id': pid,
                'player': group['player'].iloc[-1],
                'season': season,
                'team': last_team,
                'all_star': 0,   
                'g': 0,
            }
            all_rows.append(pd.DataFrame([blank_row]))
    filled_df = pd.concat(all_rows, ignore_index=True)
    filled_df = filled_df.sort_values(['player_id', 'season']).reset_index(drop=True)
    return filled_df
def get_proximity_to_prime(age):
    if 27 <= age <= 31:
        return 1
    elif age < 27:
        distance_from_prime = (27 - age) / 100
        return max(0, 1 - (distance_from_prime * 4)) # scale scores upwards as players approach their primes
    else:  # age > 31
        distance_from_prime = (31 - age)/100
        return -(max(0, 1 - (distance_from_prime * 10))) # scale score lower as players age past 31
def get_conference(team, season):
    """Returns 'East' or 'West' based on team abbreviation and season."""
    
    
    eastern_conference = ['BOS', 'BRK', 'NYK', 'PHI', 'TOR', 'CHI', 'CLE', 'DET', 'IND', 'MIL', 'ATL', 'CHA', 'WAS','NJN','ORL','MIA','CHH','NOH','CHO']
    western_conference = ['GSW', 'LAC', 'LAL', 'SAC', 'POR', 'DEN', 'UTA', 'MIN', 'NOP', 'OKC', 'DAL', 'HOU', 'SAS','VAN','SEA','NOK','PHO','MEM']
        
 
            
    # New Orleans (complex history)
    if season > 2004:
        if team == 'NOH':
            return 'West'
    if team in eastern_conference:
        return 'East'
    if team in western_conference:
        return 'West'
            
    return None

def display_all_star_team(east_df, west_df, model_results,num_of_selections=12, cols=['player', 'team']):
    
    min_season = 2000
    max_season = int(max(east_df['season'].max(), west_df['season'].max()))
    
    def plot_for_season(season):
        # --- Select top 12 predicted from each conference ---
        east_top = east_df[(east_df['season'] == season) & 
                           (east_df['predicted_all_star'] == 1)].head(num_of_selections)
        west_top = west_df[(west_df['season'] == season) & 
                           (west_df['predicted_all_star'] == 1)].head(num_of_selections)
        all_preds = pd.concat([east_top, west_top], ignore_index=True)

        # Identify true next-year All-Stars (for missed cases)
        true_east = east_df[(east_df['season'] == season) & 
                            (east_df['next_yr_all_star'] == 1)]
        true_west = west_df[(west_df['season'] == season) & 
                            (west_df['next_yr_all_star'] == 1)]
        all_true = pd.concat([true_east, true_west], ignore_index=True)

        # --- Determine categories ---
        correct = all_preds[all_preds['next_yr_all_star'] == 1]
        incorrect = all_preds[all_preds['next_yr_all_star'] == 0]
        missed = all_true[~all_true['player'].isin(all_preds['player'])]

        # --- Prepare display rows ---
        rows, row_colors, text_colors = [], [], []

        # Header
        rows.append([f"Predictions for {season + 1} season", ""])
        row_colors.append(["#FFFFFF", "#FFFFFF"])
        text_colors.append(["black", "black"])

        def add_rows(df, color, text_color="black"):
            for _, r in df.iterrows():
                player = str(r.get(cols[0], '') or '')
                team = str(r.get(cols[1], '') or '')
                rows.append([player, team])
                row_colors.append([color, color])
                text_colors.append([text_color, text_color])

        # Add each group
        add_rows(correct, "#C6F6C6")   # green
        add_rows(incorrect, "#FFB3B3") # red

        # --- Build Figure with Two Axes ---
        fig, (ax_table, ax_metrics) = plt.subplots(1, 2, figsize=(12, 8))
        ax_table.axis("off")
        ax_metrics.axis("off")

        # --- Left: Prediction Table ---
        table = ax_table.table(
            cellText=rows,
            cellLoc='left',
            colLabels=cols,
            loc='center',
            cellColours=row_colors
        )

        for i, color_row in enumerate(row_colors):
            for j in range(len(color_row)):
                table[(i, j)].get_text().set_color(text_colors[i][j])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.8, 1.5)

        # Legend
        legend_items = [
            mpatches.Patch(color='#C6F6C6', label=' Correct Prediction'),
            mpatches.Patch(color='#FFB3B3', label=' Incorrect Prediction')
        ]

        ax_table.legend(handles=legend_items, loc='lower right',  bbox_to_anchor=(1.35, -0.05), fontsize=9)

        # --- Right: Model Performance Metrics as a Table ---
        metrics_data = []
        for metric in ['ndcg@12', 'f1@12']:
            mean_val = model_results.loc['mean', metric]
            std_val = model_results.loc['std', metric]
            metrics_data.append([metric, f"{mean_val:.3f} ± {std_val:.3f}"])

        metrics_table = ax_metrics.table(
            cellText=metrics_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='center'
        )
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(12)
        metrics_table.scale(1, 2)  # make it taller

        ax_metrics.set_title("Model Summary", fontsize=14, fontweight='bold')
        ax_metrics.axis("off")

        plt.tight_layout()
        plt.show()
    interact(plot_for_season,season=Dropdown(options=list(range(min_season, max_season + 1)),value=max_season,description="Season"))
def display_top_guesses(east_df, west_df, num_of_selections=12, cols=['player', 'team']):
    """Display top predicted All-Stars from each conference acrofss all seasons."""
      
    min_season = 2000
    max_season = int(max(east_df['season'].max(), west_df['season'].max()))
    def plot_for_season(season):
       
        east_top = east_df[(east_df['season'] == season) & 
                           (east_df['predicted_all_star'] == 1)].head(num_of_selections)
        west_top = west_df[(west_df['season'] == season) & 
                           (west_df['predicted_all_star'] == 1)].head(num_of_selections)
        all_preds = pd.concat([east_top, west_top], ignore_index=True)

       
        true_east = east_df[(east_df['season'] == season) & 
                            (east_df['next_yr_all_star'] == 1)]
        true_west = west_df[(west_df['season'] == season) & 
                            (west_df['next_yr_all_star'] == 1)]
        all_true = pd.concat([true_east, true_west], ignore_index=True)

      
        correct = all_preds[all_preds['next_yr_all_star'] == 1]
        incorrect = all_preds[all_preds['next_yr_all_star'] == 0]
        missed = all_true[~all_true['player'].isin(all_preds['player'])]
      
        rows, row_colors, text_colors = [], [], []

       
        rows.append([f"Predictions for {season + 1} season", ""])
        row_colors.append(["#FFFFFF", "#FFFFFF"])
        text_colors.append(["black", "black"])

        def add_rows(df, color, text_color="black"):
            for _, r in df.iterrows():
                player = str(r.get(cols[0], '') or '')
                team = str(r.get(cols[1], '') or '')
                rows.append([player, team])
                row_colors.append([color, color])
                text_colors.append([text_color, text_color])

       
        add_rows(correct, "#C6F6C6") 
        add_rows(incorrect, "#FFB3B3") 

       
        fig, (ax_table, ax_metrics) = plt.subplots(1, 2, figsize=(12, 8))
        ax_table.axis("off")
        ax_metrics.axis("off")

      
        table = ax_table.table(
            cellText=rows,
            cellLoc='left',
            colLabels=cols,
            loc='center',
            cellColours=row_colors
        )

        for i, color_row in enumerate(row_colors):
            for j in range(len(color_row)):
                table[(i, j)].get_text().set_color(text_colors[i][j])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.8, 1.5)

     
        legend_items = [
            mpatches.Patch(color='#C6F6C6', label=' Correct Prediction'),
            mpatches.Patch(color='#FFB3B3', label=' Incorrect Prediction')
        ]

        ax_table.legend(handles=legend_items, loc='lower right',  bbox_to_anchor=(1.35, -0.05), fontsize=9)
        metrics_table = ax_metrics.table(
            cellText=[['All stars captured', f"{(len(correct))} out of 24"]],
            loc='center',
            cellLoc='center'
        )
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(12)
        metrics_table.scale(1, 2) 
        ax_metrics.set_title(f"Models top {num_of_selections * 2 } predictions", fontsize=14, fontweight='bold')
        ax_metrics.axis("off")

        plt.tight_layout()
        plt.show()
    interact(plot_for_season,season=Dropdown(options=list(range((min_season), max_season + 1)),value=max_season,description="Season"))

