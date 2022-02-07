# ## Sentiment Analysis on Grandma's WhatsApp chat

import pandas as pd

chat_txt = '_chat.txt'
with open(chat_txt, encoding='utf-8') as file:
    content = file.read()
import re
import pandas as pd
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
i=0
def remove_date(text):
    newones = []
    stopword = (stopwords.words('english'))
    querywords = text.split()

    resultwords  = [word for word in querywords if word.lower() not in stopword]
    text = ' '.join(resultwords)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.lower()
    #text = re.sub('\\n*', '', text)
    text = text.split(": " or "\n")
   
    text = [i.split('  ') for i in text]
    while('' in text):
        text.remove('')
    return text

data_use = remove_date(content)

data_use = data_use.pop()
index1 = []
i=0
for x in data_use:
    index1.append(i)
    i = i+1
dict = {'Index':index1, 'Messages':data_use[0]}
df1 = pd.DataFrame({'Index':index1})
df2 = pd.DataFrame({'Messages':data_use})
df_tot = [df1, df2]
df_fin = pd.concat(df_tot, axis=1)

data = df_fin
data.head()
data = data.dropna()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Messages"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Messages"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Messages"]]

print("\n**positive sentiment:**")
print(data[data["Positive"] !=0].head(2))
print("\n**negative sentiment:**")
print(data[data["Negative"] !=0].head(2))

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        return ("Positive")
    elif (b>a) and (b>c):
        return ("Negative")
    elif (c>a) and (c>a):
        return ("Neutral")
    else:
        return ("Inconclusive")
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

answer = sentiment_score(x, y, z)
lists = [x,y,z]
cats = ['Positive', 'Negative', 'Neutral']
tot_df = pd.DataFrame()
tot_df['Category'] = cats
tot_df['Data'] = lists

print("The result from sentiment analysis of whatsapp chat is: " + answer)
tot_df.head()

import matplotlib.pyplot as plt
plt.figure(figsize=[9,9])
my_labels = 'Positive %','Negative %','Neutral %'
my_colors = ['lightgreen','pink','beige']
my_explode = (0.025, 0.025, 0.025)
plt.pie(tot_df['Data'], labels=my_labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors, explode=my_explode)
plt.title('Sentiment Analysis')
plt.ylabel('')
plt.show()
