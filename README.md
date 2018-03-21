# GenSimilarArtists
Find similar artists based on gensim and topic modelling.

## The Idea

While experimenting with Topic Modelling and NLP I had the idea of a tool that can be used like this.

* _Tool_: "Hey there. Tell me some names of music artists and I fetch their Wikipedia page and make some NLP things with it."

* _User_: "Here are some artist names. What now?"

* _Tool_ (fetching data and building a model of the Wikipedia content): "Name another artist."

* _User_: "Timbaland."

* _Tool_: "Alrighty. I fetch his Wikipedia page and compare it to the pages I already know. Then I return you the name of the most fitting pages. One moment."

* _User_: "..."

* _Tool_: "The most fitting artists seem to be: `Tweet (p=0.9369953274726868), Ms. Jade (p=0.9053862690925598), Timbaland & Magoo (p=0.9013492465019226), Static Major (p=0.8887353539466858), Missy Elliott (p=0.8651071786880493), D.O.E. (p=0.7915201187133789), Ginuwine (p=0.7398850917816162), Lyrica Anderson (p=0.7208942174911499), Nicole Wray (p=0.7165670394897461)`"

* _User_: "what."

* _Tool_: "What?"

* _User_: "That actually works."

Since Wikipedia does not always return the right article for an misleading artist name I started to use the Last.FM API and got my articles from there.

## Music Artist Example

Take a look at the Jupyter notebook [music_artist.ipynb](https://github.com/mymindwentblvnk/gensimilarity/blob/master/music_artists.ipynb) to see how this works. You have to provide Last.FM API information in settings.py to make it run yourself. You can use my pickled model which saves you a lot of time.

## Movie Example

tbd.

## Summary
I guess it works. There were no super embarrassing results. If the source data would cover every music genre the results would be much better I guess. 

Genres I listen to a lot create very good results. 

Is this thing practically? I would say no, because it takes hours to build the model from source data with ~5200 artists. And the models, if persisted on disk, take up a good amount of space. 34 MB for the model.


But what is charming is that I can apply this tool to topics I don't even know anything about. I just need to fetch source data e. g. from Wikipedia and make it work. With this I can ask for similar books or movies only by their summary.
Also very good is that I don't have to parse the source data for keywords in any way. The gensim model I have chosen works really good.
