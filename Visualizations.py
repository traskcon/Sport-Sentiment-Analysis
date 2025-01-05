import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("Sentiment-Data.csv")
team_data = data[["team","predicted sentiment"]]
league_data = data[["league","predicted sentiment"]]
team_sentiment = team_data.groupby(["team"]).mean()
team_sentiment.sort_values(by=["predicted sentiment"], ascending=False, inplace=True)
league_sentiment = league_data.groupby(["league"]).mean()
league_sentiment.sort_values(by=["predicted sentiment"], ascending=False, inplace=True)
team_sentiment.to_csv("team-sentiment.csv")
'''data.boxplot(by="league",column=["predicted sentiment"])
plt.title("Sentiment Distribution by League")
plt.suptitle("")
plt.ylabel("Predicted Sentiment")
plt.show()'''

originals = pd.read_csv("reddit_data.csv")
data.sort_values(by=["predicted sentiment"], inplace=True, ascending=False)
top_positives = data.iloc[:5]["id"].to_list()
top_negatives = data.iloc[-5:]["id"].to_list()
print(originals.loc[originals["id"].isin(top_positives)])
print(originals.loc[originals["id"].isin(top_negatives)])