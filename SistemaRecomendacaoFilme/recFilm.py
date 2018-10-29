# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:01:10 2018

@author: Daniel Lucas
"""
# Importação do  Pandas
import pandas as pd

# carregar os dados do cvs
metadata = pd.read_csv('tmdb_5000_movies.csv', low_memory=False)

# Print the first three rows
metadata.head(3)
#  Classificação ponderada (WR) = (vv + m.R) + (mv + m.C)
#  Onde,
#
#  v é o número de votos para o filme;
#  m é o mínimo de votos necessários para ser listado no gráfico;
#  R é a classificação média do filme; E
#  C é a média de votos em todo o relatório

# calcular a média de votos de todos filmes
C = metadata['vote_average'].mean()
print(C)

#  Calcular o número mínimo de votos necessários para estar no gráfico, m
m = metadata['vote_count'].quantile(0.90)
print(m)

# Filtre todos os filmes qualificados em um novo DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape

#Função que calcula a classificação ponderada de cada filme
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Cálculo baseado na fórmula do IMDB
    return (v/(v+m) * R) + (m/(m+v) * C)

# Defina um novo recurso 'score' e calcule seu valor com `weighted_rating ()
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

##Cortar filmes com base na pontuação calculada acima
q_movies = q_movies.sort_values('score', ascending=False)

# mostrar os melhores 15 primeiros filmes 
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)

#Imprima os resumos dos primeiros 5 filmes.
metadata['overview'].head()

#Importar TfIdfVectorizer de scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Defina um objeto Vectorizer TF-IDF. Remova todas as palavras de parada em inglês, como "the", "a"
tfidf = TfidfVectorizer(stop_words='english')

## Substitua NaN por uma string vazia
metadata['overview'] = metadata['overview'].fillna('')

#Construir a matriz TF-IDF necessária, ajustando e transformando os dados
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output a forma de tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Calcule a matriz de similaridade de cosseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construir um mapa inverso de índices e títulos de filmes
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Função que leva no título do filme como entrada e saída de filmes mais semelhantes
def get_recommendations(title, cosine_sim=cosine_sim):
    # Obter o índice do filme que corresponde ao título
    idx = indices[title]

    # Obtenha as pontuações de similaridade entre todos os filmes com esse filme
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Classifique os filmes com base nas pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obter as pontuações dos 10 filmes mais semelhantes
    sim_scores = sim_scores[1:11]

  
    # Obter os índices do filme
    movie_indices = [i[0] for i in sim_scores]

    # Retorna os 10 filmes mais parecidos
    return metadata['title'].iloc[movie_indices]
    # Ler o nome do filme que "foi visto"
    get_recommendations('The Dark Knight Rises')

#get_recommendations('The Godfather') 