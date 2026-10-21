import sqlite3
from datetime import datetime

import streamlit as st
import spacy
from LeIA import SentimentIntensityAnalyzer  # VADER oficial para Português do Brasil

# -----------------------------------------------------------------------------
# 1. Configurações da Página e Recursos
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PLN PT-BR & Análise de Sentimento",
    page_icon="🇧🇷",
    layout="wide"
)

@st.cache_resource
def load_spacy_pt():
    """Carrega o modelo do spaCy para Português."""
    return spacy.load("pt_core_news_sm")

@st.cache_resource
def load_leia():
    """Carrega o analisador de sentimento LeIA (VADER adaptado para PT-BR)."""
    return SentimentIntensityAnalyzer()

nlp = load_spacy_pt()
analyzer = load_leia()

# -----------------------------------------------------------------------------
# 2. Banco de Dados SQLite3
# -----------------------------------------------------------------------------
DB_NAME = "historico_ptbr.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT NOT NULL,
            texto_original TEXT NOT NULL,
            sentimento TEXT NOT NULL,
            compound REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def salvar_analise(data_hora, texto, sentimento, compound):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analises (data_hora, texto_original, sentimento, compound)
        VALUES (?, ?, ?, ?)
    """, (data_hora, texto, sentimento, compound))
    conn.commit()
    conn.close()

def buscar_historico():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT data_hora, texto_original, sentimento, compound FROM analises ORDER BY id DESC")
    dados = cursor.fetchall()
    conn.close()
    return dados

init_db()

# -----------------------------------------------------------------------------
# 3. Processamento de Linguagem Natural e Sentimento em PT-BR
# -----------------------------------------------------------------------------
def processar_texto_pt(texto):
    """
    Fatia (tokeniza) o texto em Português e remove stop words, pontuações e numerais.
    """
    doc = nlp(texto)
    tokens_filtrados = [
        token.text for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num and not token.is_space
    ]
    return doc, tokens_filtrados

def analisar_sentimento_pt(texto_pt):
    """
    Calcula a polaridade diretamente em Português usando o LeIA.
    Retorna o rótulo do sentimento e o valor de compound.
    """
    scores = analyzer.polarity_scores(texto_pt)
    compound = scores['compound']
    
    # Classificação baseada no limiar padrão do VADER
    if compound >= 0.05:
        sentimento = "Positivo"
    elif compound <= -0.05:
        sentimento = "Negativo"
    else:
        sentimento = "Neutro"
        
    return sentimento, compound

# -----------------------------------------------------------------------------
# 4. Interface do Usuário (Streamlit)
# -----------------------------------------------------------------------------
st.title("🇧🇷 Processamento de Texto e Análise de Sentimento em PT-BR")
st.markdown("Aplicação utilizando **spaCy (pt_core_news_sm)**, **LeIA (VADER para PT-BR)** e **SQLite3**.")

tab1, tab2 = st.tabs(["📝 Processar Texto", "📊 Histórico SQLite3"])

with tab1:
    texto_input = st.text_area(
        "Digite seu texto em português:",
        height=130,
        placeholder="Ex: Eu estou muito feliz com esse resultado!"
    )
    
    if st.button("Analisar e Salvar", type="primary"):
        if not texto_input.strip():
            st.warning("Por favor, digite um texto para continuar.")
        else:
            # 1. Fatiamento e remoção de ruídos com spaCy
            doc, tokens_limpos = processar_texto_pt(texto_input)
            
            # 2. Análise de sentimento nativa em PT-BR
            sentimento, compound = analisar_sentimento_pt(texto_input)
            
            # 3. Obtenção de data e hora
            data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 4. Salvar no SQLite3
            salvar_analise(data_hora_atual, texto_input, sentimento, compound)
            st.success("Análise armazenada com sucesso no SQLite3!")
            
            # Exibição dos resultados
            col_esq, col_dir = st.columns(2)
            
            with col_esq:
                st.subheader("✂️ Fatiamento e Filtro (spaCy PT)")
                st.write(f"**Total de tokens originais:** {len(doc)}")
                st.write("**Tokens filtrados (sem stop words/pontuação):**")
                st.json(tokens_limpos)
                
            with col_dir:
                st.subheader("📊 Métrica de Sentimento")
                st.metric(label="Classificação", value=sentimento)
                st.metric(label="Métrica Compound", value=f"{compound:.4f}")
            
            # Análise detalhada dos tokens
            with st.expander("🔍 Ver análise morfossintática dos tokens (spaCy)"):
                tabela_tokens = [
                    {
                        "Token": token.text,
                        "Lema (Raiz)": token.lemma_,
                        "Classe Gramatical (POS)": token.pos_,
                        "É Stop Word?": token.is_stop,
                        "É Pontuação?": token.is_punct
                    }
                    for token in doc
                ]
                st.dataframe(tabela_tokens, use_container_width=True)

with tab2:
    st.subheader("📜 Histórico de Análises Salvas")
    dados = buscar_historico()
    
    if dados:
        historico_df = [
            {
                "Data/Hora": row[0],
                "Texto Input": row[1],
                "Sentimento": row[2],
                "Compound": row[3]
            }
            for row in dados
        ]
        st.dataframe(historico_df, use_container_width=True)
    else:
        st.info("Nenhum registro encontrado no banco de dados.")