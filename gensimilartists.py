import re
import pickle

import pylast

import nltk  # See https://pythonspot.com/nltk-stop-words/
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import gensim
from gensim.models import TfidfModel

import settings


MODEL_PICKLE_PATH = 'music/model.pickel'
MAPPING_PICKLE_PATH = 'music/mapping.pickel'

ARTIST_NAMES_PATH = 'music/artist_names.txt'


def get_artist_names():
    with open(ARTIST_NAMES_PATH, 'r') as f:
        artist_names = f.readlines()
    artist_names = [a.strip() for a in artist_names]
    return artist_names


class TheTfidfModel(object):

    def __init__(self, dct, corpus, model, tfidf_corpus, lsi, index):
        self.dct = dct
        self.corpus = corpus
        self.model = model
        self.corpus_tfidf = tfidf_corpus
        self.lsi = lsi
        self.index = index


class LastFmClient(object):

    def __init__(self):
        import settings
        api_key=settings.API_KEY
        api_secret=settings.SHARED_SECRET
        username=settings.USERNAME
        password_hash=pylast.md5(settings.PW)

        self.client = pylast.LastFMNetwork(api_key=api_key,
                                           api_secret=api_secret,
                                           username=username,
                                           password_hash=password_hash)

    def get_artist_info(self, artist_name):
        try:
            artist = self.client.get_artist(artist_name)
            name = artist.get_name()
            bio = artist.get_bio_content()

            if 'There are at least' not in bio:  # No perfect match found
                return {
                    'artist_name': name,
                    'bio': bio
                }
        except:
            return None


class GenSimilArtists(object):

    def __init__(self, from_scratch=False):
        self.last_fm_client = LastFmClient()
        self.mapping = self._create_mapping(from_scratch)
        self.model = self._create_model(from_scratch)

    def _create_mapping(self, from_scratch):

        if not from_scratch:
            try:  # to load model from disk
                with open(MAPPING_PICKLE_PATH, 'rb') as pickle_in:
                    mapping = pickle.load(pickle_in)
                    return mapping
            except FileNotFoundError:
                pass

        print("Creating mapping.")
        mapping_dict = {}
        doc_number = 0

        artist_names = get_artist_names()
        for index, artist_name in enumerate(artist_names, 1):
            print("{}/{}".format(index, len(artist_names)), end='\r')
            info = self.last_fm_client.get_artist_info(artist_name)

            if info:
                mapping_dict[doc_number] = {
                    'artist_name': info['artist_name'],
                    'bio': self._clean_text(info['bio']),
                }
                doc_number += 1

        with open(MAPPING_PICKLE_PATH, 'wb') as pickle_out:
            pickle.dump(mapping_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

        return mapping_dict

    def _create_model(self, from_scratch):

        if not from_scratch:
            try:  # to load model from disk
                with open(MODEL_PICKLE_PATH, 'rb') as pickle_in:
                    model = pickle.load(pickle_in)
                    return model
            except FileNotFoundError:
                pass

        print("Creating model.")
        dataset = list()
        for doc_number in range(0, len(self.mapping.keys())):
            dataset.append(self.mapping[doc_number]['bio'])

        dct = gensim.corpora.Dictionary(dataset)
        corpus = [dct.doc2bow(line) for line in dataset]
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus = tfidf_model[corpus]
        lsi = gensim.models.LsiModel(tfidf_corpus, id2word=dct, num_topics=50, power_iters=4)
        index = gensim.similarities.MatrixSimilarity(lsi[tfidf_corpus])

        model = TheTfidfModel(dct=dct,
                              corpus=corpus,
                              model=tfidf_model,
                              tfidf_corpus=tfidf_corpus,
                              lsi=lsi,
                              index=index)

        with open(MODEL_PICKLE_PATH, 'wb') as pickle_out:
            pickle.dump(model, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

        return model


    def get_similar_artists(self, artist_name, n=5):
        info = self.last_fm_client.get_artist_info(artist_name)
        if info:
            bio = self._clean_text(info['bio'])

            vec_bow = self.model.dct.doc2bow(bio)
            vec_lsi = self.model.lsi[self.model.model[vec_bow]]
            sims = self.model.index[vec_lsi]

            result = list()

            d = dict(enumerate(sims))
            top_n = sorted(d, key=d.get, reverse=True)[:n]

            for doc_number in top_n:
                result.append({
                    'propability': sims[doc_number],
                    'artist_name': self.mapping[doc_number]['artist_name']
                })

            return result
        else:
            print("No Last.FM bio for {} found.".format(artist_name))

    def _clean_text(self, text):
        text = self._remove_html_tags(text)
        text = self._remove_control_chars(text)
        text = self._remove_stop_words(text)
        return text

    def _remove_html_tags(self, text):
        cleaner = re.compile('<.*?>')
        return re.sub(cleaner, '', text)


    def _remove_control_chars(self, text):
        text = text.replace('\n','')
        text = text.replace('\t','')
        return text


    def _remove_stop_words(self, text):
        stop_words = list()
        stop_words.extend(stopwords.words('english'))

        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        result = []
        for word in words:
            if word not in stop_words:
                result.append(word)
        return list(result)


if __name__ == '__main__':
    gsa = GenSimilArtists()
    name = "Timbaland"

    result = gsa.get_similar_artists(name, n=10)

    if result:
        print("The most fitting artists for", name)
        print(20*"*")
        for r in result:
            print(r['artist_name'], "(p={})".format(r['propability']))
