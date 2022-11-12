# # Download stopwords list
import datetime
import datetime as dt
import io
import json
import logging
import os
import os.path
import pickle

import bs4
import joblib
import nltk
import numpy as np
import pandas as pd
import requests
import scipy
import scipy.sparse
import sqlalchemy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# from imblearn.ensemble import EasyEnsembleClassifier

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

LISTINGS_FILE = '../data/listings.zip'
POSTERS_FILE = '../data/posters.zip'
CATS_FILE = '../data/cats.zip'
X_FILE = '../data/X'
y_FILE = '../data/y'
SVD_FILE = '../data/svd.npz'
RESULTS_FILE = '../data/results.pickle'
TFIDF_VECTORIZOR_FILE = '../data/tfidf_vectorizer'
MEDIAN_FILE = '../data/price_median'

logging.basicConfig()

D_THRESH = 100
FIRST_N_CATS_VALUES = [20, 50, 100]
L_LEVEL = logging.WARNING


class Timerek:
    _t = datetime.datetime.now()

    @classmethod
    def ping(cls, text=''):
        t1 = cls._t
        t2 = cls._t = datetime.datetime.now()
        return f'{text}: {t2 - t1}'


def add_results(name, accuracy: float, cr: str):
    d1 = {}
    for line in cr.splitlines():
        s = line.split()
        d1['accuracy'] = accuracy
        if len(s) == 5:
            name, precision, recall, f1_score, support = s
            d1[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support
            }
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, 'rb') as f:
            d = pickle.load(f)
    else:
        d = {}
    d[name] = d1
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(d, f)


def configure_logging():
    # create logger with 'spam_application'
    logger = logging.getLogger('dmed')
    logger.setLevel(L_LEVEL)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('dmed.log')
    fh.setLevel(L_LEVEL)
    # create console handler with a higher log level
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger


def load_posters(engine) -> pd.DataFrame:
    logger.debug(f'Loading posters.')

    posters = pd.read_sql_table('posters', engine)
    posters.set_index('id', verify_integrity=True, inplace=True)
    # wywalam kolumny:
    posters.drop(inplace=True, columns=['search_date',
                                        'current_page',
                                        'complete',
                                        'lock_date',
                                        'lock_id',
                                        'status',
                                        'ignore_phrases',
                                        'max_price',
                                        'search_interval'])
    # zmiana nazw
    posters.rename(columns={'name': 'posters_name',
                            'type': 'posters_type',
                            'url': 'posters_url'}, inplace=True)
    b = io.StringIO()
    posters.info(buf=b)
    logger.debug(f'Done loading posters:\n{b.getvalue()}\n')
    return posters


def load_cats(engine) -> pd.DataFrame:
    def merge_rec(r: pd.Series, c: pd.DataFrame) -> pd.Series:
        """
        Spłaszczenie drzewa kategorii
        """
        p_id = r.parent_id
        # do tego będę doklejał
        name = r['name']
        # tak długa jak parent nie jest nan i jest w indeksie (czasem brakuje, np 164)
        while pd.notna(p_id) and p_id in c.index:
            parent_cat = c.loc[p_id]
            name = f'{parent_cat["name"]} -> {name}'
            # podstawiam znalezionego parenta i od nowa
            p_id = parent_cat.parent_id
        r['cat_name_full'] = name
        return r

    logger.debug('Loading cats')
    cats = pd.read_sql_table('categories', engine)
    cats.set_index('id', verify_integrity=True, inplace=True)
    # zamiana parent_id na int
    cats['parent_id'] = cats.parent_id.convert_dtypes(convert_integer=True)
    # spłaszczenie drzewa kategorii do pola cat_name_full
    cats = cats.apply(func=merge_rec, args=(cats,), axis=1)
    b = io.StringIO()
    cats.info(buf=b)
    logger.debug(f'Done loading cats:\n{b.getvalue()}\n')
    return cats


def join_posters(posters: pd.DataFrame, cats: pd.DataFrame) -> pd.DataFrame:
    posters_joined = posters.join(cats, on='category_id', rsuffix='_cats')
    posters_joined.drop(columns=['name', 'parent_id'], inplace=True)
    return posters_joined


def load_posters_history(engine, posters) -> pd.DataFrame:
    logger.debug('Loading posters history')

    posters_history: pd.DataFrame = pd.read_sql_table('posters_history', engine)
    # wybieram tylko dotmed
    listings: pd.DataFrame = posters_history[posters_history.type == 'dotmed']
    listings.set_index('id', verify_integrity=True, inplace=True)
    listings = listings.join(posters, on='poster_id', rsuffix='_cat')
    # usuwam puste kolumny
    listings.dropna(axis=1, how='all', inplace=True)
    # usuwam niepuste kolumny - UWAGA na type - usuwam bo wcześniej == 'dotmed'
    listings.drop(columns=['hash', 'poster_id', 'type', 'date', 'img', 'deleted', 'deleted_date',
                           'category_id', 'posters_type', 'posters_name'], inplace=True)

    # w description_second jest json parametrów listingu
    # Zamieniam puste descriptiony na {}
    listings.loc[listings.description_second.str.contains('null'), 'description_second'] = '{}'
    # Zamiana napisu na słownik
    d1: pd.Series = listings.description_second.apply(json.loads)
    d2: pd.Series = d1.apply(lambda d: {f'd_{k.lower()}': v for k, v in d.items()})
    # klucze słownika -> kolumny
    a1: pd.DataFrame = pd.json_normalize(d2)
    # usuwam prawie puste
    a1.dropna(axis=1, thresh=D_THRESH, inplace=True)
    # łączę
    listings = listings.join(a1)
    listings.drop(columns=['description_second'], inplace=True)
    b = io.StringIO()
    listings.info(buf=b, verbose=True)
    logger.debug(f'Done loading listings:\n{b.getvalue()}\n')
    return listings


def load_all_from_db(engine) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    Timerek.ping()
    posters = load_posters(engine)
    cats = load_cats(engine)
    posters_joined = join_posters(posters, cats)
    listings = load_posters_history(engine, posters_joined)
    logger.info(
        f'Done loading all from DB. {len(cats)} cats, {len(posters)} posters, {len(listings)} listings. Time: {Timerek.ping()}.')
    return cats, posters_joined, listings


def save_pickle(cats: pd.DataFrame, posters: pd.DataFrame, listings: pd.DataFrame):
    Timerek.ping()
    cats.to_pickle(CATS_FILE)
    posters.to_pickle(POSTERS_FILE)
    listings.to_pickle(LISTINGS_FILE)
    logger.debug(f'Done pickling data in {Timerek.ping()}.')


def load_pickle() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    Timerek.ping()
    cats = pd.read_pickle(CATS_FILE)
    posters = pd.read_pickle(POSTERS_FILE)
    listings = pd.read_pickle(LISTINGS_FILE)
    logger.debug(f'Done unpickling data in {Timerek.ping()}.')
    return cats, posters, listings


def is_up_to_date(file) -> bool:
    if os.path.isfile(file):
        ts = os.path.getmtime(file)
        dt_f = dt.datetime.fromtimestamp(ts)
        dt_t = dt.datetime.today()
        return dt_f.year == dt_t.year and dt_f.day == dt_t.day and dt_f.day == dt_t.day
    else:
        return False


def are_up_to_date():
    return is_up_to_date(CATS_FILE) and is_up_to_date(POSTERS_FILE) and is_up_to_date(LISTINGS_FILE)


def optional_read_DB(force_read=False, prohibit_read=False):
    if prohibit_read or are_up_to_date() and not force_read:
        logger.info('DB data up to date, loading...')
        return *load_pickle(), False
    elif force_read or not are_up_to_date():
        engine = sqlalchemy.create_engine(
            "mariadb+mariadbconnector://projekty_medic:Medic2022!@vps1829.uphost.pro:3306/projekty_medic")
        logger.info(f'Loading data from DB, force_read={force_read}...')
        cats, posters, listings = load_all_from_db(engine)
        save_pickle(cats, posters, listings)
        return cats, posters, listings, True
    else:
        raise ValueError('force_read and probibit_read at the same time?')


def get_top_n_cats(listings: pd.DataFrame, n: int) -> (pd.DataFrame, np.ndarray,):
    logger.debug(f'Getting top {n} cats...')
    lvc = listings.posters_url.value_counts()
    listings_big = listings[listings.posters_url.isin(lvc[:n].index)].copy()
    wielkosci = listings_big.posters_url.value_counts().values
    logger.info(f'Done getting top {n} cats...')
    return listings_big, wielkosci


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


def preprocess(listings: pd.DataFrame,
               tfidf_vectorizer: TfidfVectorizer = None,
               no_cats=666,
               auction_parameters=True,
               ) -> (scipy.sparse.spmatrix, np.ndarray):
    logger.debug('Starting preprocessing...')

    logger.debug(f'generating y and removing y columns')

    if 'posters_url' in listings.columns:
        # filling price with medians calculated per label only if labels in input
        listings["price"] = listings.groupby("posters_url")['price'].transform(
            lambda x: x.fillna(0 if x.isnull().all() else x.median()))
        logger.debug(f'Filled prices na with median in {Timerek.ping()}.')

        logger.debug(f'Converting urls to dmed cats')
        listings['dmed_cat'] = convert_urls_to_dmed_cats(listings.posters_url)
        logger.debug(f'Converting urls to dmed cats done.')
        y = listings.dmed_cat.copy().values
        listings.drop(columns=['posters_url', 'url', 'cat_name_full', 'dmed_cat'],
                      inplace=True)
    else:
        y = None

    # usuwam puste kolumny
    listings.dropna(axis=1, how='all', inplace=True)
    logger.debug(f'Cleared empty columns in {Timerek.ping()}.')

    # normalizacja nazw kolumn
    listings.columns = listings.columns.str.lower().str.replace(r'\W', '_', regex=True).str.replace('_+', '_',
                                                                                                    regex=True).str.strip(
        '_')
    logger.debug(f'Fixed column names in {Timerek.ping()}.')

    listings['description'] = listings.description.str.replace(r'[\W_]', ' ', regex=True) \
        .str.replace(r'\s+', ' ', regex=True) \
        .str.lower(). \
        str.replace('x ray', 'xray')
    listings['title'] = listings.title.str.lower()
    logger.debug(f'Cleared description in {Timerek.ping()}.')

    stop_words = set(stopwords.words('english'))
    more = '''used
    like new
    tested
    in good condition
    good condition
    has been fully tested
    fully tested
    please contact us if you have more questions
    please contact us
    contact us if you have more questions
    more questions
    excellent condition
    excellent
    concerns
    do not hesitate to contact me thank you
    thank you
    contact me
    contact us'''
    stop_words = stop_words | set(more.splitlines())
    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))
    if not tfidf_vectorizer:
        tfidf_vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
        tfidf_vectorizer.fit(listings.description.values)
        logger.debug(f'Fitted Vectorizer in {Timerek.ping()}')
        with open(f'{TFIDF_VECTORIZOR_FILE}_{no_cats}.pickle', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

    # wektoryzacja pola description
    vectorised_descriptions = tfidf_vectorizer.transform(listings.description.values)
    logger.debug(f'Transformed descriptions in {Timerek.ping()}')
    vectorised_descriptions = vectorised_descriptions.tocsc()
    logger.debug(f'Vectorized description To CSC  in {Timerek.ping()}')

    # wektoryzacja pola title
    titles = listings.title.copy().values
    logger.debug(f'Seperated titles in {Timerek.ping()}')
    vectorised_titles = tfidf_vectorizer.transform(titles)
    logger.debug(f'Transformed titles in {Timerek.ping()}')
    vectorised_titles = vectorised_titles.tocsc()
    logger.debug(f'Vectorized title To CSC  in {Timerek.ping()}')

    prices = listings.price.copy().values.reshape(-1, 1)
    logger.debug(f'Seperated prices in {Timerek.ping()}')

    listings.drop(columns=['description', 'price', 'title'], inplace=True)
    logger.debug('Seperated labels and deleted columns')

    if auction_parameters:
        # one hot encoding reszty pól
        Timerek.ping()
        vectorised_rest = pd.get_dummies(listings,
                                         prefix_sep='_',
                                         drop_first=True,
                                         sparse=True,
                                         dtype=np.float32).sparse.to_coo()
        logger.debug(f'Got dummies in {Timerek.ping()}')
        logger.debug(
            f'Getting ready to hstack: {[e.shape for e in (prices, vectorised_titles, vectorised_descriptions, vectorised_rest)]}')
        X = scipy.sparse.hstack((prices, vectorised_titles, vectorised_descriptions, vectorised_rest))
    else:
        logger.debug(
            f'Getting ready to hstack: {[e.shape for e in (prices, vectorised_titles, vectorised_descriptions)]}')
        X = scipy.sparse.hstack((prices, vectorised_titles, vectorised_descriptions))
    logger.debug(f'HStacked in {Timerek.ping()}')
    logger.info('Done preprocessing...')
    return X, y


def read_X_y(no_cats,
             force_read=False,
             prohibit_read=False,
             force_preprocess=False,
             auction_parameters=True,
             ) -> (
        scipy.sparse.spmatrix, np.ndarray, np.ndarray):
    cats, posters_joined, listings, is_new = optional_read_DB(force_read=force_read, prohibit_read=prohibit_read)
    if is_new or force_preprocess:
        logger.debug(f'Starting generating data for {FIRST_N_CATS_VALUES} top cat counts.')
        for no in FIRST_N_CATS_VALUES:
            logger.debug(f'Starting for {no} cats...')
            listings_tmp, wielkosci = get_top_n_cats(listings, no)
            X, y = preprocess(listings_tmp, no_cats=no, auction_parameters=auction_parameters)
            scipy.sparse.save_npz(f'{X_FILE}_{no}.npz', X)
            np.savez_compressed(f'{y_FILE}_{no}.npz', y=y, wielkosci=wielkosci)
            if no_cats == no:
                logger.debug(f'copying data for returning for {no} cats...')
                X_ret = X
                y_ret = y
            logger.debug(f'Done for {no} cats...')
    else:
        logger.debug(f'Up-to-date data. Loading for {no_cats} top categories...')
        loaded1 = np.load(f'{y_FILE}_{no_cats}.npz', allow_pickle=True)
        y_ret = loaded1['y']
        X_ret = scipy.sparse.load_npz(f'{X_FILE}_{no_cats}.npz')
        wielkosci = loaded1['wielkosci']
    return X_ret, y_ret, wielkosci


def load_svd(cat_n, svd_n):
    data = np.load(SVD_FILE, allow_pickle=True)
    y = data[f'y_svd{cat_n}']
    X = data[f'X_{cat_n}_svd{svd_n}']
    with open(f'..\data\svd_X_{cat_n}_svd{svd_n}.joblib', 'rb') as f:
        svd = joblib.load(f)
    with open(f'..\data\s_scaler_X_{cat_n}_svd{svd_n}.joblib', 'rb') as f:
        s_scaler = joblib.load(f)
    return X, y, svd, s_scaler


def convert_urls_to_dmed_cats(url_series: pd.Series, domyslna='Inna', join_str=': '):
    return url_series.replace({url: get_dmed_cat_breadcrumb(url) for url in url_series.unique()})


def get_dmed_cat_breadcrumb(url, domyslna='Inna', join_str=': '):
    try:
        t = requests.get(url).text
        b = bs4.BeautifulSoup(t, features="html.parser")
        wynik = b.find('ul', attrs={'aria-label': 'breadcrumb'}).find_all('span')[1:]
        wynik1 = [w.text for w in wynik]
        wynik2 = join_str.join(wynik1)
        logger.debug(f'Converted url: "{url}"')
        logger.debug(f'To category  : "{wynik2}"')
        return wynik2

    except Exception as e:
        logger.warning(f'Nie udało się wczytać katoegorii: {url}')
        logger.warning(f'Błąd: {e.__class__.__name__}: {e}, args: {e.args}')
        return domyslna


class MedPredictor():
    def __init__(self, clf, tfidf: TfidfVectorizer, *steps):
        self._clf = clf
        self._tfidf = tfidf
        self._steps = steps

    def predict(self, price: float, title: str, description: str):
        df = pd.DataFrame({
            'price': [price],
            'title': [title],
            'description': [description]
        })

        X_my, _ = preprocess(df, auction_parameters=False, tfidf_vectorizer=self._tfidf)
        for step in self._steps:
            X_my = step.transform(X_my)
        return self._clf.predict(X_my)


logger = configure_logging()

if __name__ == '__main__':
    # print(s_scaler, type(s_scaler))
    # X = s_scaler.transform(X)
    # # best_p = {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}
    # # clf = KNeighborsClassifier(n_jobs=32, **best_p)
    # clf = MLPClassifier(
    #     max_iter=400,
    #     alpha=0.02,
    #     early_stopping=True,
    #     batch_size=800,
    #     learning_rate='adaptive',
    #     verbose=True,
    #     warm_start=True,
    #     learning_rate_init=0.02,
    #     hidden_layer_sizes=(99, 49),
    # )
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    # print(Timerek.ping('Fitting'))
    # clf.fit(X_train, y_train)
    # print(Timerek.ping('Predicting'))
    # y_pred = clf.predict(X_test)
    # print(Timerek.ping('Reporting'))
    # print(classification_report(y_test, y_pred, target_names=clf.classes_))
    # with open('../data/best_mlp.joblib', 'wb') as f:
    #     joblib.dump(clf, f)

    price = 10.0
    title = 'HITACHI MR-QFC-A15 MRI Coil'
    description = '''1x Hitachi Large Quad Flex Coil
MN#: MR-QFC-A15
S/N: 039129
Est Dimensions: 77x17x2
Used Cosmetic Condition: Good
Total # of Pieces: 1

MPN: MR-QFC-A115
Information
Buyers Premium: 18% with 3% cash discount

Sales Tax: 6.625%

Inspection
Wednesday, November 16th
Time: 11AM - 12:00PM EST Inspect online on our Facebook Page from 11am-12pm.

Equipment Removal
The Removal period will be from Friday, November 18th to Friday, December 02nd. Items that are either a) not removed or b) have not been scheduled for shipping via DOTmed Logistics by the end of the Removal Period will be charged $100 per day per lot.

New Storage Option for Winners: Store your stuff with us till the next event for $50 per pallet space. Click here to find out more.

Address:
3 South Middlesex,
Monroe Township, NJ 08831

For domestic and international shipping info, click here.

Have questions for us? Call us at 212-742-1200 ext 298 or Email us'''
    with open(f'..\data\svd_X_50_svd50.joblib', 'rb') as f:
        svd = joblib.load(f)
    with open(f'..\data\s_scaler_X_50_svd50.joblib', 'rb') as f:
        s_scaler = joblib.load(f)

    with open('../data/best_mlp.joblib', 'rb') as f:
        clf = joblib.load(f)
    with open('../data/tfidf_vectorizer_50.pickle', 'rb') as f:
        tfidf = pickle.load(f)
    p = MedPredictor(clf, tfidf, svd, s_scaler)
    y_my = p.predict(price, title, description)
    print(y_my)
    y_my = p.predict(price, title, description)
    print(y_my)
    y_my = p.predict(price, title, description)
    print(y_my)
    y_my = p.predict(price, title, description)
    print(y_my)
