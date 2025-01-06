'''
This script scrapes each day's reddit sentiment data, labels it using the pre-trained model, then updates the visualizations
'''

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Model import preprocess_data
import pandas as pd
import keras
import pickle
import numpy as np
import datetime as dt
import re
import csv
import praw
from tqdm import tqdm

## Function Definitions ########################################
def get_date(submission):
    # Extract the timestamp of a reddit comment/post
    time = submission.created
    return dt.datetime.fromtimestamp(time)

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\uFFFD" #replacement character
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def make_safe_string(text):
    text = remove_emojis(text)
    # Remove newlines, tabs, and commas
    text = text.replace("\n", "").replace("\t","").replace(",","")
    return text

def most_recent_posts():
    # Create a dictionary of each teams name and its most recent post date
    last_posts = dict()
    data = pd.read_csv("reddit_data.csv")
    for team in pd.unique(data["team"]):
        team_posts = data[data["team"] == team]
        last_posts[team] = max(team_posts["date"])
    return last_posts

def extract_reddit_comments(league, reddit, last_posts):
    # Scrape the 20 newest posts from each team since last update
    filename = "reddit_data.csv"
    with open(filename, "a+", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        for team in league_teams[league]:
            team_post_date = last_posts[team]
            subreddit = reddit.subreddit(team)
            for post in tqdm(subreddit.new(limit=20)):
                if get_date(post) > team_post_date:
                    all_comments = post.comments.list()
                    # comment.body returns the raw text of the comment
                    # comment.id is a unique identifier for the comment
                    for comment in all_comments:
                        try:
                            data = [team, comment.id, get_date(comment), make_safe_string(comment.body), league]
                            writer.writerow(data)
                        except:
                            continue

## END FUNCTION DEFINITIONS #########################################

## Scrape Reddit Comments ###########################################

reddit = praw.Reddit(
    client_id = "",
    client_secret = "",
    user_agent = "Sentiment-Analysis",
    ratelimit_seconds = 600
)
# Without username and password this is just a read-only agent

league_teams = {"NFL":["detroitlions","patriots","losangelesrams","eagles","greenbaypackers","cowboys","steelers","49ers","seahawks","minnesotavikings","chibears","falcons","kansascitychiefs","browns","ravens","denverbroncos","nygiants","washingtonnfl","buffalobills","saints","buccaneers","miamidolphins","bengals","nyjets","panthers","azcardinals","texans","colts","tennesseetitans","chargers","jaguars","raiders"],
                "NHL":["leafs","hawks","detroitredwings","penguins","canucks","bostonbruins","flyers","habs","sanjosesharks","caps","newyorkislanders","wildhockey","rangers","devils","bluejackets","tampabaylightning","edmontonoilers","coloradoavalanche","stlouisblues","goldenknights","anaheimducks","winnipegjets","predators","canes","sabres","calgaryflames","losangeleskings","ottawasenators","dallasstars","coyotes","floridapanthers","SeattleNHL"],
                "NBA":["warriors","lakers","bostonceltics","torontoraptors","sixers","chicagobulls","rockets","NYKnicks","clevelandcavs","Thunder","MkeBucks","mavericks","NBASpurs","timberwolves","washingtonwizards","UtahJazz","ripcity","suns","kings","heat","denvernuggets","AtlantaHawks","GoNets","OrlandoMagic","DetroitPistons","LAclippers","pacers","charlotteHornets","NOLAPelicans","memphisgrizzlies"],
                "MLB":["MiamiMarlins","azdiamondbacks","coloradoRockies","buccos","tampabayrays","KcRoyals","TexasRangers","OaklandAthletics","motorcitykitties","Reds","clevelandGuardians","Brewers","angelsbaseball","Nationals","whitesox","minnesotatwins","Padres","Mariners","orioles","NewYorkMets","cHIcubs","cardinals","phillies","Astros","SFGiants","Braves","Torontobluejays","Dodgers","redsox","NYYankees"],
                "NWSL":["AngelcityFc","BayFc","RedStars","Dash","Kccurrent","GothamFc","Nccourage","ReignFc","OrlandoPride","Thorns","RacingLouisvilleFc","SanDiegoWaveFc","UtahRoyalsFc","WashingtonSpirit"],
                "WNBA":["AtlantaDream","chicagoSky","connecticutSun","indianafever","NYLiberty","washingtonmystics","dallaswings","VegasAces","LASparks","MinnesotaLynx","PHXMercury","seattlestorm"]}

league_subreddits = {"NFL":["nfl"],
                     "NHL":["hockey","nhl"],
                     "NBA":["nba"],
                     "MLB":["mlb"],
                     "NWSL":["nwsl"],
                     "WNBA":["wnba"]}

last_posts = most_recent_posts()
for league in league_teams.keys():
    extract_reddit_comments(league, reddit, last_posts)

data = pd.read_csv("reddit_data.csv")    
cleaned_data = preprocess_data(data)

## END DATA COLLECTION #######################################################

## Calculate Sentiment Using Pre-Trained Model ###############################
model = keras.saving.load_model("glove_model.keras")
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
vectorizer = keras.layers.TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(cleaned_data["comment"])
vectorizer.set_weights(from_disk['weights'])

model_data = np.array(vectorizer(np.array([[s] for s in cleaned_data["comment"]])))

data["predicted sentiment"] = model.predict(model_data)

## END SENTIMENT PREDICTION ##################################################

# TODO: Create interactive visualizations via Dash/Plotly