import pandas as pd
from musixmatch import Musixmatch
#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


musixmatch = Musixmatch('883f168836a4c0ae244f06359b397a45')

analyser = SentimentIntensityAnalyzer()

df=pd.read_csv('Spotify_final.csv',index_col=0)

sentiment_list = []
sentiment_score_list = []

for i in df[['Track name', 'Artist name']].values:

    try:
        song = musixmatch.matcher_lyrics_get(i[1], i[0])
        song = song['message']['body']['lyrics']['lyrics_body']
        sentiment_score = analyser.polarity_scores(song)

        if sentiment_score['compound'] >= 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Positive'
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Neutral'
        elif sentiment_score['compound'] <= -0.05:
            sentiment_percentage = sentiment_score['compound']
            sentiment = 'Negative'

        sentiment_list.append(sentiment)
        sentiment_score_list.append((abs(sentiment_percentage) * 100))

    except:
        sentiment_list.append('None')
        sentiment_score_list.append(0)

#aa=musixmatch.matcher_lyrics_get('The One', 'Kodaline')
#ab=aa['message']['body']['lyrics']['lyrics_body']
#ac=analyser.polarity_scores(ab)
#print(ac)


df['sentiment']=sentiment_list
df['sentiment score']=sentiment_score_list

df.to_csv('Spotify_Sentiment_Analysis.csv')