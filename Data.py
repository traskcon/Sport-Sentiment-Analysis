import praw
import datetime as dt

def get_date(submission):
    time = submission.created
    return dt.datetime.fromtimestamp(time)

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

for team in league_teams["NFL"]:
    # Obtain subreddit instance
    subreddit = reddit.subreddit(team)
    # Obtain 100 newest posts
    for post in subreddit.new(limit=100):
        all_comments = post.comments.list()
        print(get_date(all_comments[0]))
    break