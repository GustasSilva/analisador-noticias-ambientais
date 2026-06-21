"""
Módulo principal do Analisador de Notícias Ambientais

Este módulo implementa a classe AnalisadorNoticias que realiza:
- Pré-processamento de texto
- Classificação de sentimentos e assuntos
- Coleta e análise de notícias da web
- Geração de relatórios
"""

import pandas as pd
import os
import joblib
import numpy as np
import re
import requests
import lxml.html.clean
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from newspaper import Article
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk

# Baixa as stopwords em português (silencioso; só baixa se ainda não existir)
nltk.download('stopwords', quiet=True)

class AnalisadorNoticias:
    """Classe principal que implementa o analisador de notícias"""
    
    def __init__(self):
        """Inicializa o analisador com configurações padrão"""
        self.stop_words = stopwords.words('portuguese')  # Lista de stopwords
        self.arquivo_csv = "dataset_noticias.csv"  # Arquivo para armazenar os dados
        self.carregar_dados()  # Carrega dados existentes
        self.inicializar_modelos()  # Inicializa os modelos de ML
    
    def carregar_dados(self):
        """
        Carrega os dados do arquivo CSV ou cria um DataFrame vazio se o arquivo não existir
        
        Realiza:
        - Leitura do arquivo CSV se existir
        - Conversão da coluna de data para datetime
        - Cria estrutura vazia se for um novo dataset
        """
        if os.path.exists(self.arquivo_csv):
            self.df = pd.read_csv(self.arquivo_csv)
            
            # Converter para datetime especificando UTC para timezone-aware
            self.df['Data_Cadastro'] = pd.to_datetime(
                self.df['Data_Cadastro'],
                utc=True,  # Adicione este parâmetro
                dayfirst=False,
                format='mixed'
            ).dt.tz_localize(None)  # Remove o timezone após conversão
        else:
            # Cria um DataFrame vazio com as colunas necessárias
            self.df = pd.DataFrame(columns=[
                "Tipo_Noticia", "Texto", "Classificação", "Percentual_Negativa",
                "Percentual_Positiva", "Percentual_Neutra", "Data_Cadastro"
            ])

    def preprocessar_texto(self, texto):
        """
        Pré-processa o texto para análise
        
        Parâmetros:
        texto (str): Texto a ser processado
        
        Retorna:
        str: Texto processado
        
        Realiza:
        - Conversão para minúsculas
        - Normalização de caracteres acentuados
        - Remoção de pontuação
        """
        texto = texto.lower()
        texto = re.sub(r"[áàâãä]", "a", texto)  # Normaliza 'a'
        texto = re.sub(r"[éèêë]", "e", texto)    # Normaliza 'e'
        texto = re.sub(r"[íìîï]", "i", texto)    # Normaliza 'i'
        texto = re.sub(r"[óòôõö]", "o", texto)    # Normaliza 'o'
        texto = re.sub(r"[úùûü]", "u", texto)    # Normaliza 'u'
        texto = re.sub(r"[ç]", "c", texto)       # Normaliza 'ç'
        texto = re.sub(r"[^\w\s]", "", texto)    # Remove pontuação
        return texto
    
    def inicializar_modelos(self):
        """
        Inicializa os modelos de machine learning e imprime métricas de avaliação
        """
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words=self.stop_words)
        self.modelo_sentimento_arquivo = "modelo_sentimento.pkl"
        self.modelo_assunto_arquivo = "modelo_assunto.pkl"
        
        if not self.df.empty:
            X = self.vectorizer.fit_transform(self.df["Texto"])
            
            if 'Classificação' in self.df.columns:
                # Modelo de análise de sentimento
                self.label_encoder_sentimento = LabelEncoder()
                y_sentimento = self.label_encoder_sentimento.fit_transform(self.df["Classificação"])
                
                # Divisão treino-teste
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X, y_sentimento, test_size=0.2, random_state=42, stratify=y_sentimento)
                
                # Treinamento
                self.modelo_sentimento = SVC(kernel="rbf", C=2, gamma="scale", probability=True)
                self.modelo_sentimento.fit(X_train_s, y_train_s)
                
                # Avaliação
                y_pred_s = self.modelo_sentimento.predict(X_test_s)
                
                print("\n=== Métricas do Modelo de Sentimento ===")
                print(f"Acurácia: {accuracy_score(y_test_s, y_pred_s):.2f}")
                print(f"F1-Score (média macro): {f1_score(y_test_s, y_pred_s, average='macro'):.2f}")
                print("\nRelatório de Classificação:")
                print(classification_report(y_test_s, y_pred_s, 
                                         target_names=self.label_encoder_sentimento.classes_))
                print("\nMatriz de Confusão:")
                print(confusion_matrix(y_test_s, y_pred_s))
                
                joblib.dump(self.modelo_sentimento, self.modelo_sentimento_arquivo)
            
            if 'Tipo_Noticia' in self.df.columns:
                # Modelo de classificação de assunto
                self.label_encoder_assunto = LabelEncoder()
                y_assunto = self.label_encoder_assunto.fit_transform(self.df["Tipo_Noticia"])
                
                # Divisão treino-teste
                X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
                    X, y_assunto, test_size=0.2, random_state=42, stratify=y_assunto)
                
                # Treinamento
                self.modelo_assunto = SVC(kernel="linear", C=1, gamma="scale")
                self.modelo_assunto.fit(X_train_a, y_train_a)
                
                # Avaliação
                y_pred_a = self.modelo_assunto.predict(X_test_a)
                
                print("\n=== Métricas do Modelo de Assunto ===")
                print(f"Acurácia: {accuracy_score(y_test_a, y_pred_a):.2f}")
                print(f"F1-Score (média macro): {f1_score(y_test_a, y_pred_a, average='macro'):.2f}")
                print("\nRelatório de Classificação:")
                print(classification_report(y_test_a, y_pred_a,
                                         target_names=self.label_encoder_assunto.classes_))
                print("\nMatriz de Confusão:")
                print(confusion_matrix(y_test_a, y_pred_a))
                
                joblib.dump(self.modelo_assunto, self.modelo_assunto_arquivo)
        else:
            # Inicializa encoders com categorias padrão se não houver dados
            self.label_encoder_sentimento = LabelEncoder()
            self.label_encoder_sentimento.fit(["Positiva", "Negativa", "Neutra"])
            self.label_encoder_assunto = LabelEncoder()
            self.label_encoder_assunto.fit(["Queimadas", "Desmatamento", "Poluição", "Outros"])
    
    def analisar_noticia(self, texto):
        """
        Analisa uma notícia e retorna classificação de assunto e sentimento
        
        Parâmetros:
        texto (str): Texto da notícia a ser analisada
        
        Retorna:
        dict: Dicionário com resultados da análise contendo:
          - texto_processado
          - assunto
          - sentimento
          - percentuais (Positiva, Negativa, Neutra)
        """
        texto_processado = self.preprocessar_texto(texto)
        texto_vetorizado = self.vectorizer.transform([texto_processado])
        
        # Classificação do assunto
        assunto = self.label_encoder_assunto.inverse_transform(
            self.modelo_assunto.predict(texto_vetorizado)
        )[0]
        
        # Análise de sentimento com probabilidades
        proba = self.modelo_sentimento.predict_proba(texto_vetorizado)[0]
        sentimento = self.label_encoder_sentimento.classes_[np.argmax(proba)]
        
        # Calcula percentuais para cada classe de sentimento
        percentuais = {
            classe: proba[i] * 100
            for i, classe in enumerate(self.label_encoder_sentimento.classes_)
        }
        return {
            "texto_processado": texto_processado,
            "assunto": assunto,
            "sentimento": sentimento,
            "percentuais": {
                "Positiva": percentuais.get("Positiva", 0.0),
                "Negativa": percentuais.get("Negativa", 0.0),
                "Neutra": percentuais.get("Neutra", 0.0)
            }
        }

    def coletar_e_analisar_url(self, url: str, limite_caracteres=600):
        """
        Coleta e analisa uma notícia a partir de uma URL
        
        Parâmetros:
        url (str): URL da notícia
        limite_caracteres (int): Limite de caracteres para análise (default: 600)
        
        Realiza:
        - Download e parse do artigo
        - Limitação do tamanho do texto
        - Análise do conteúdo
        - Salvamento dos resultados
        """
        try:
            # Coletar e processar o artigo usando a biblioteca newspaper
            artigo = Article(url, language='pt')
            artigo.download()
            artigo.parse()

            texto = artigo.text.strip()
            if not texto:
                raise ValueError("Nenhum texto foi encontrado na URL.")

            # Limita o tamanho do texto para análise
            texto_limitado = texto[:limite_caracteres]

            # Realiza a análise do texto
            resultado = self.analisar_noticia(texto_limitado)

            # Prepara os dados para salvar
            dados = {
                "Tipo_Noticia": resultado["assunto"],
                "Texto": texto_limitado,
                "Classificação": resultado["sentimento"],
                "Percentual_Negativa": resultado["percentuais"].get("Negativa", 0.0),
                "Percentual_Positiva": resultado["percentuais"].get("Positiva", 0.0),
                "Percentual_Neutra": resultado["percentuais"].get("Neutra", 0.0),
                "Data_Cadastro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "URL": url
            }

            # Salva em arquivo CSV, adicionando a coluna URL se necessário
            arquivo_csv = "noticias_analisadas.csv"
            novo_df = pd.DataFrame([dados])

            if os.path.exists(arquivo_csv):
                df_existente = pd.read_csv(arquivo_csv)
                if "URL" not in df_existente.columns:
                    df_existente["URL"] = ""  # Adiciona a coluna se não existir
                df_completo = pd.concat([df_existente, novo_df], ignore_index=True)
            else:
                df_completo = novo_df

            df_completo.to_csv(arquivo_csv, index=False)
            print("Notícia analisada e salva com sucesso.")

        except Exception as e:
            print(f"Erro ao coletar/analisar URL: {e}")

    def extrair_texto_da_url(self, url):
        """
        Extrai o texto principal de uma URL usando BeautifulSoup
        
        Parâmetros:
        url (str): URL para extração de conteúdo
        
        Retorna:
        str: Texto extraído
        
        Realiza:
        - Requisição HTTP para obter o conteúdo
        - Remoção de scripts e estilos
        - Extração de texto de parágrafos
        """
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts e estilos para obter apenas o texto relevante
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Junta os textos de todos os parágrafos
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        
        if not text:
            raise ValueError("Não foi possível extrair conteúdo da URL.")
        
        return text
    
    def salvar_noticia(self, dados):
        """
        Salva uma notícia analisada no dataset
        
        Parâmetros:
        dados (dict): Dicionário com os resultados da análise
        
        Realiza:
        - Cria novo registro com os dados
        - Atualiza o DataFrame
        - Salva no arquivo CSV
        - Re-inicializa os modelos com os novos dados
        """
        novo_registro = {
            "Tipo_Noticia": dados["assunto"],
            "Texto": dados["texto_processado"],
            "Classificação": dados["sentimento"],
            "Percentual_Negativa": dados["percentuais"]["Negativa"],
            "Percentual_Positiva": dados["percentuais"]["Positiva"],
            "Percentual_Neutra": dados["percentuais"]["Neutra"],
            "Data_Cadastro": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Adiciona o novo registro e salva
        self.df = pd.concat([self.df, pd.DataFrame([novo_registro])], ignore_index=True)
        self.df.to_csv(self.arquivo_csv, index=False)
        self.inicializar_modelos()  # Re-treina os modelos com os novos dados

    def gerar_dados_relatorio(self, data_inicio=None, data_fim=None):
        """
        Prepara os dados para geração de relatórios
        
        Parâmetros:
        data_inicio (str): Data de início do período (opcional)
        data_fim (str): Data de fim do período (opcional)
        
        Retorna:
        tuple: (relatorio_exibicao, df_exibicao, resumo_sentimentos)
        
        Realiza:
        - Filtragem por período
        - Agregação de dados por data
        - Preparação de dados para exibição
        """
        try:
            df = self.df.copy()

            # Converter a coluna de data já tratando timezones
            df['Data_Cadastro'] = pd.to_datetime(
                df['Data_Cadastro'],
                utc=True
            ).dt.tz_localize(None)  # Remove o timezone

            # Filtra por período se especificado
            if data_inicio:
                data_inicio = pd.to_datetime(data_inicio).tz_localize(None)
                df = df[df['Data_Cadastro'] >= data_inicio]
            if data_fim:
                data_fim = pd.to_datetime(data_fim).tz_localize(None)
                df = df[df['Data_Cadastro'] <= data_fim]

            # Debug: mostra informações sobre o filtro
            print("Data Início:", data_inicio)
            print("Data Fim:", data_fim)
            print("Data_Cadastro após conversão:", df['Data_Cadastro'].head())
            print("Intervalo de datas no DataFrame:", df['Data_Cadastro'].min(), df['Data_Cadastro'].max())

            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Agrupa os dados por data
            relatorio = df.groupby((df['Data_Cadastro'].dt.date)).agg({
                'Tipo_Noticia': 'count',  # Conta notícias por dia
                'Percentual_Positiva': 'mean',  # Média dos percentuais
                'Percentual_Negativa': 'mean',
                'Percentual_Neutra': 'mean'
            }).rename(columns={'Tipo_Noticia': 'Total_Noticias'}).dropna()

            # Cria resumo por tipo de notícia e classificação
            resumo_sentimentos = df.groupby(['Tipo_Noticia', 'Classificação']).size().unstack(fill_value=0)

            # Prepara cópias formatadas para exibição
            relatorio_exibicao = relatorio.copy()
            relatorio_exibicao.index = pd.to_datetime(relatorio_exibicao.index).strftime('%Y-%m-%d')

            df_exibicao = df.copy()
            df_exibicao['Data_Cadastro'] = df_exibicao['Data_Cadastro'].dt.strftime('%Y-%m-%d')

            return relatorio_exibicao, df_exibicao, resumo_sentimentos

        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()