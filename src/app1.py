import pandas as pd
import streamlit as st
import random
import pickle
import openpyxl

if 'slownik_procedur' not in st.session_state:
    try:
        with open('kat_procedura.pickle', 'rb') as f:
            st.session_state['slownik_procedur'] = pickle.load(f)
    except OSError as e:
        st.session_state['slownik_procedur'] = {}


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


@st.cache
def przewiduj(tytul):
    print('Przewiduję')
    if tytul:
        return random.choice(['Tomograf', 'USG', 'Inne'])
    else:
        return 'Nieznane'


st.title('Przewidywanie typu urządzenia')
st.subheader('Metoda: KNearest Neighbors')
st.text_input('Tytuł urządzenia', key='tytul')
st.slider(label='Cena urządzenia', min_value=1, max_value=1000000, key='cena')
st.text_area('Opis urządzenia', key='opis')
st.text_input('Wynik', value=przewiduj(st.session_state.tytul), disabled=True, key='wynik')
st.multiselect('Procedury',
               options=load_icd(),
               default=load_procedury(),
               on_change=save_procedury,
               key='procedury',
               help='')
