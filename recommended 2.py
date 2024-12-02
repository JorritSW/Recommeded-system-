# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:40:11 2024

@author: Jorrit Wierda
"""

# Systeem bibliotheken
import re
import unicodedata
import itertools

# Bibliotheek voor bestandsmanipulatie
import pandas as pd
import numpy as np
import pandas

# Datavisualisatie
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import pyplot as plt

# Machine learning
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Laad de bestanden
spotify_songs_path = 'spotify_songs.csv'
streaming_history_0_path = 'StreamingHistory_music_0.csv'
streaming_history_1_path = 'StreamingHistory_music_1.csv'

# Lees de bestanden
spotify_songs_df = pd.read_csv("E:/Data/spotify_songs.csv")
streaming_history_0_df = pd.read_csv("E:/Data/StreamingHistory_music_0.csv")
streaming_history_1_df = pd.read_csv("E:/Data/StreamingHistory_music_1.csv")

# Combineer de streaming geschiedenis
streaming_history_df = pd.concat([streaming_history_0_df, streaming_history_1_df], ignore_index=True)

# Normaliseer de kolomnamen
spotify_songs_df['track_artist'] = spotify_songs_df['track_artist'].str.lower()
spotify_songs_df['track_name'] = spotify_songs_df['track_name'].str.lower()

streaming_history_df['artistName'] = streaming_history_df['artistName'].str.lower()
streaming_history_df['trackName'] = streaming_history_df['trackName'].str.lower()

# Zoek overeenkomende nummers
matched_songs = spotify_songs_df.merge(
    streaming_history_df,
    left_on=['track_name', 'track_artist'],
    right_on=['trackName', 'artistName'],
    how='inner'
)

# Opslaan als een nieuw CSV-bestand
output_path = 'matched_songs_dataset.csv'
matched_songs.to_csv(output_path, index=False)

# Configuratie voor breedte en indeling van grafieken
sns.set_theme(style='whitegrid')
palette='viridis'

# Waarschuwingen uitschakelen
import warnings
warnings.filterwarnings("ignore")

# Bibliotheekversies laden
import watermark

# Database
df = pd.read_csv("E:/Data/spotify_songs.csv")
df

# Eerste 5 gegevens bekijken
df.head()

# Laatste 5 gegevens bekijken
df.tail()

# Gegevensinfo
df.info()

# Gegevenstypen
df.dtypes

# Rijen en kolommen bekijken
df.shape

# Gegevens kopiëren
data = df.copy()

# Frequentie van categorische variabelen
print()
print(df['playlist_genre'].value_counts())

# Top 10 meest voorkomende artiesten
print()
print(df['track_artist'].value_counts().head(10)) 

from sklearn.model_selection import train_test_split, GridSearchCV

# De dataset splitsen in training en test
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

print("Training bekijken x_train", train_data.shape)
print("Test bekijken test_data", test_data.shape)

# Voorgedefinieerd lied om aanbevelingen op te baseren
# Vervang door het lied waarmee je wilt starten
default_song_name = "Someone You Loved - Future Humans Remix"  

def recommend_songs(df, model, interaction_matrix, song_name=default_song_name, k=10):
    # Zoek de afspeellijst(en) die het voorgedefinieerde lied bevatten
    song_playlists = df[df['track_name'].str.contains(song_name, case=False, na=False)]['playlist_id'].unique()
    
    if len(song_playlists) == 0:
        print("Lied niet gevonden in de dataset.")
        return
    
    # Haal het ID van de eerste afspeellijst op waar het lied werd gevonden
    playlist_id = song_playlists[0]
    
    # Unieke afspeellijst-ID's en hun corresponderende indices ophalen in interaction_matrix
    unique_playlists = df['playlist_id'].unique()
    
    if playlist_id not in unique_playlists:
        print("Afspeellijst-ID niet gevonden in unieke afspeellijsten.")
        return
    
    playlist_index = np.where(unique_playlists == playlist_id)[0][0]
    
    # Vind vergelijkbare afspeellijsten met behulp van KNN
    try:
        # Als interaction_matrix een DataFrame is, gebruik .iloc om de rij op te halen
        if isinstance(interaction_matrix, pd.DataFrame):
            distances, indices = model.kneighbors(interaction_matrix.iloc[playlist_index].values.reshape(1, -1), n_neighbors=k+1)
        else:
            distances, indices = model.kneighbors(interaction_matrix[playlist_index].reshape(1, -1), n_neighbors=k+1)
    except IndexError:
        print("Afspeellijstindex is buiten bereik in de interactiematrix.")
        return
    
    # Verwijder de index van de originele afspeellijst
    similar_playlists = indices.flatten()[1:]  # De eerste overslaan, want dat is de originele afspeellijst
    
    # Identificeer aan te bevelen liedjes
    original_playlist_tracks = set(df[df['playlist_id'] == playlist_id]['track_id'])
    recommended_tracks = set()

    for idx in similar_playlists:
        similar_playlist_id = unique_playlists[idx]
        similar_playlist_tracks = set(df[df['playlist_id'] == similar_playlist_id]['track_id'])
        recommended_tracks.update(similar_playlist_tracks - original_playlist_tracks)

    # Aanbevolen liedjes weergeven
    if recommended_tracks:
        recommended_tracks_info = df[df['track_id'].isin(recommended_tracks)][['track_name', 'track_artist']].drop_duplicates()
        print("Aanbevolen liedjes")
        print(recommended_tracks_info)
    else:
        print("Geen nieuwe liedjes om aan te bevelen.")

# Database 
def load_data(path):
    return pd.read_csv(path)

# Gebruik de functie om data te laden
data = load_data(output_path)

# Bekijk de eerste paar rijen
print(data.head())

# 1. Selecteer de relevante kenmerken voor clustering
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

X = data[features]

# Data schalen
from sklearn.preprocessing import StandardScaler

# Kenmerken schalen
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bekijken
scaler

from sklearn.cluster import KMeans

# Het optimale aantal clusters bepalen met behulp van de elleboogmethode
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Elleboogplot weergeven
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Aantal Clusters')
plt.ylabel('Som van Kwadratische Afstanden (Inertia)')
plt.title('Elleboogmethode voor Bepalen van het Aantal Clusters')
plt.grid(False)
plt.show()

# 4. K-Means toepassen met het gekozen aantal clusters
# Stel dat het optimale aantal clusters 5 is (pas aan op basis van de elleboogplot)
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
kmeans

# Totaal aantal clusters bekijken
data.cluster_kmeans.value_counts()

# Aantal liedjes in elke cluster
cluster_counts = data['cluster_kmeans'].value_counts().sort_values(ascending=False)

# Het percentage van elke cluster berekenen
total_songs = cluster_counts.sum()
cluster_percentages = (cluster_counts / total_songs) * 100

# De verdeling van clusters plotten met percentages
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='deep')
plt.xlabel('Cluster')
plt.ylabel('Aantal Liedjes')
plt.title('Verdeling van Liedjes per Cluster')

# Gegevenslabels bovenop de balken toevoegen
for index, value in enumerate(cluster_counts):
    plt.text(index, value + 100, f'{value} ({cluster_percentages[index]:.1f}%)', ha='center', fontsize=12)

plt.grid(False)
plt.show()

from sklearn.decomposition import PCA

# 2. PCA-analyse uitvoeren
# Initialiseer het PCA-model en specificeer het aantal componenten om te reduceren (hier 2 componenten).
pca = PCA(n_components=2)

# Pas PCA toe op de geschaalde kenmerken en transformeer de gegevens naar de nieuwe 2-dimensionale ruimte.
X_pca = pca.fit_transform(X_scaled)

# Toepassen van K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['cluster_kmeans'] = clusters

# Plotten van de clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], 
                y=X_pca[:, 1], 
                hue=data['cluster_kmeans'], 
                palette='viridis', 
                alpha=0.6, 
                edgecolor='k')

# Toevoegen van de clustercentra
centers = kmeans.cluster_centers_
# Projecteren van de centra naar de 2D PCA-ruimte
centers_pca = pca.transform(centers)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='red', 
            s=200, 
            marker='X', 
            label='Clustercentra')

plt.title('Clusters na toepassing van K-Means')
plt.xlabel('Hoofcomponent 1')
plt.ylabel('Hoofcomponent 2')
plt.legend()
plt.grid(False)
plt.show()

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['cluster_kmeans'], palette='viridis', s=70, alpha=0.7)

# Plotten van de clustercentra
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroiden')

plt.title('Visualisatie van clusters na PCA')
plt.xlabel('Hoofcomponent 1')
plt.ylabel('Hoofcomponent 2')
plt.legend(title='Clusters')
plt.grid(False)
plt.show()

# Aannemende dat je het gemiddelde van kenmerken voor cluster 0 wilt berekenen
cluster_1_features = data[data['cluster_kmeans'] == 0][features].mean()
print("Gemiddelde kenmerken voor cluster 0:")
print(cluster_1_features)
print()

# Aannemende dat je het gemiddelde van kenmerken voor cluster 1 wilt berekenen
cluster_2_features = data[data['cluster_kmeans'] == 1][features].mean()
print("Gemiddelde kenmerken voor cluster 1:")
print(cluster_2_features)
print()

# Aannemende dat je het gemiddelde van kenmerken voor cluster 2 wilt berekenen
cluster_3_features = data[data['cluster_kmeans'] == 2][features].mean()
print("Gemiddelde kenmerken voor cluster 2:")
print(cluster_3_features)
print()

# Aannemende dat je het gemiddelde van kenmerken voor cluster 3 wilt berekenen
cluster_4_features = data[data['cluster_kmeans'] == 3][features].mean()
print("Gemiddelde kenmerken voor cluster 3:")
print(cluster_4_features)
print()

def recommend_songs_by_cluster_kmeans(song_name, data):
    # Zoeken naar het geselecteerde nummer in de dataset, hoofdletterongevoelig en omgaan met ontbrekende waarden
    selected_song = data[data['track_name'].str.contains(song_name, case=False, na=False)]
    
    # Als het nummer niet wordt gevonden, geef een melding weer en sluit de functie
    if selected_song.empty:
        print("Nummer niet gevonden.")
        return None
    
    # Ophalen van het cluster waartoe het geselecteerde nummer behoort
    cluster = selected_song['cluster_kmeans'].values[0]
    
    # Vind alle nummers die tot hetzelfde cluster behoren als het geselecteerde nummer
    recommended_songs = data[data['cluster_kmeans'] == cluster]
    
    # Het geselecteerde nummer uitsluiten van de aanbevelingen
    recommended_songs = recommended_songs[recommended_songs['track_name'] != selected_song['track_name'].values[0]]
    
    # Retourneer alleen de kolommen 'track_name' en 'track_artist', beperkt tot de top 25 aanbevelingen
    return recommended_songs[['track_name', 'track_artist']].head(30)

# Voorbeeld van gebruik
# Vervang door de naam van het nummer dat je als basis wilt gebruiken voor aanbevelingen
song_name = "Stan"  
top_recommendations_kmeans = recommend_songs_by_cluster_kmeans(song_name, data)

# Als er aanbevelingen worden gevonden, print deze
if top_recommendations_kmeans is not None:
    print("Aanbevolen muziek")
    print(top_recommendations_kmeans)
