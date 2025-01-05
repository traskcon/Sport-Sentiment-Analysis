# Sport-Sentiment-Analysis
DATA641 Final Project

Training dataset is Sentiment140: https://www.kaggle.com/datasets/kazanova/sentiment140

## Daily Workflow
The following are the actions that should run automatically each day:
1. Scrape all comments from x most recent posts on all subreddits
    * Scraper should be smart to avoid ratelimits (scrape from most recent post in existing database to present)
    * This also avoids duplicating posts in the database
2. Run the model on the new posts to update the sentiment data
3. Update plots with new data, recalculate summary statistics
4. Push updates to the website

## Development Pipeline
