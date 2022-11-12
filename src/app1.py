import os
import pickle

import joblib
import pandas as pd
import streamlit as st

from dmed import MedPredictor

if 'slownik_procedur' not in st.session_state:
    print(os.getcwd())
    try:
        with open('kat_procedura.pickle', 'rb') as f:
            st.session_state['slownik_procedur'] = pickle.load(f)
    except OSError as e:
        st.session_state['slownik_procedur'] = {}

@st.cache(allow_output_mutation=True)
def load_predictor():
    print('load predictor')
    with open(f'../data\svd_X_50_svd50.joblib', 'rb') as f:
        svd = joblib.load(f)
    with open(f'../data\s_scaler_X_50_svd50.joblib', 'rb') as f:
        s_scaler = joblib.load(f)
    with open('../data/best_mlp.joblib', 'rb') as f:
        clf = joblib.load(f)
    with open('../data/tfidf_vectorizer_50.pickle', 'rb') as f:
        tfidf = pickle.load(f)
    p = MedPredictor(clf, tfidf, svd, s_scaler)
    return p


if 'predictor' not in st.session_state:
    st.session_state['predictor'] = load_predictor()


@st.cache
def load_icd():
    df = pd.read_excel(
        'https://www.nfz.gov.pl/gfx/nfz/userfiles/_public/dla_swiadczeniodawcy/icd9/icd-9_pl_w_5.65.xlsx',
        header=[0, 1])
    df.rename(columns={'nr ': 'nr',
                       'Rozdział': 'rozdzial',
                       'Podrozdział': 'podrozdzial',
                       'Kategoria główna': 'kat_glowna',
                       'Kategoria szczegółowa': 'kat_szczegolowa',
                       'tytuł': 'tytul'}, inplace=True)
    return df.kat_glowna.tytul.dropna().unique()


def save_procedury():
    if 'wynik' in st.session_state:
        k = st.session_state.wynik
        w = st.session_state.procedury
        if st.session_state['slownik_procedur'].get(k, []) != w:
            print('saving')
            st.session_state['slownik_procedur'][k] = w
            print(st.session_state['slownik_procedur'])
            with open('kat_procedura.pickle', 'wb') as f:
                pickle.dump(st.session_state['slownik_procedur'], f)


def load_procedury():
    return st.session_state['slownik_procedur'].get(st.session_state.wynik, [])


def przewiduj(tytul, cena, opis):
    cena = float(cena)
    print('Przewiduję dla:')
    print(repr(tytul))
    print(repr(cena))
    print(repr(opis))
    r = st.session_state['predictor'].predict(float(cena), tytul, opis)
    print(f'Wynik: {r}')
    return r[0]


st.title('Przewidywanie typu urządzenia')
st.subheader('Metoda: MLP')
st.text_input('Tytuł urządzenia', key='tytul')
st.number_input(label='Cena urządzenia', min_value=1, max_value=1000000, key='cena')
st.text_area('Opis urządzenia', key='opis')

st.text_input('Wynik',
              value=przewiduj(st.session_state.tytul, st.session_state.cena, st.session_state.opis),
              disabled=True,
              key='wynik')

st.multiselect('Procedury',
               options=load_icd(),
               default=load_procedury(),
               on_change=save_procedury,
               key='procedury',
               help='')
