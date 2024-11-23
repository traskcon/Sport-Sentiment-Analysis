import praw
import datetime as dt
import pandas as pd
import csv, io
import re
from tqdm import tqdm

def get_date(submission):
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

reddit = praw.Reddit(
    client_id = "",
    client_secret = "",
    user_agent = "Sentiment-Analysis",
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


with open("reddit_data_mlb.csv", "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, delimiter=",")
    for team in league_teams["MLB"]:
        print(team)
        # Obtain subreddit instance
        subreddit = reddit.subreddit(team)
        # Obtain 100 newest posts
        for post in tqdm(subreddit.new(limit=20)):
            all_comments = post.comments.list()
            # comment.body returns the raw text of the comment
            # Use get_date function to extract timestep of comment
            # comment.id is a unique identifier for the comment
            for comment in all_comments:
                try:
                    data = [team, comment.id, get_date(comment), make_safe_string(comment.body)]
                    writer.writerow(data)
                except:
                    continue