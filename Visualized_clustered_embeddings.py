"""
Erstellen von eine Supervised oder unsupervised darstellung von einer Sammlung von Dokumente
anhand der Semantische Ähnlichkeit. 

Erzeugt eine cloud.html html Datei mit dem Plot.

Spezifikationen:
* Sprachmodell: sentence-transformers/all-MiniLM-L6-v2 
    (multilingual weil Zielanwendungsprache ggf nicht english)
* Klustering Algorithmus: Kmeans von sklearn
* NLTK für Vocabulary und häufigkeit von Wörter
* UMAP für Projektion der Embeddigns

Variablen: 
doc: Documenten Corpus
cluster: Anzahl von clustern
top_n: Wörter zur Beschreibung des clusters

April 2023
Anna Capilla Fernández
annacapilla@gmail.com

"""

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
import pandas as pd
import umap
import plotly.express as px
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
english_stop_words = stopwords.words('english')

#Laden von Daten
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
docs = fetch_20newsgroups(subset='all',  remove=('footers', 'quotes', 'headers'))['data'][:400]

#Pre-processing: typischerweise stop Wörter entfernen (und/der/die/das/Präpositionen, usw)
def clean(docs: list) -> list:
    #Preprocessing hier
    docs = [' '.join([word for word in doc_i.split() if word not in english_stop_words]) for doc_i in docs]
    return docs


#Laden des Models. Wir brauchen ein Modell der Dokumente encoden kann, ich schlage vor Sentence Transformer, es gibt davon aus mehreren versionen
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Mit dem pretrainierten Modell, generieren wir eine Darstellung jeder Dokument.
#Es handelt sich um einen Vector, was das Modell aus dem Text generiert.
clean_docs=clean(docs)
embeddings = model.encode(clean_docs)

#Nun clustern wir die Embeddings in x classen (5 in dem Beispiel).
# Das ist nicht sueprvised, dh das kriterium wonach es sortiert ist ist unbekannt und kann sich ändern
clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(embeddings)

#Wir packen alles zusammen in einem Datensatz
df = pd.DataFrame(embeddings)
df['text'] = docs
df['clean_text'] = clean_docs
df['extract']=df['text'].apply(lambda x: x[:100])
df['labels']= kmeans.labels_

#Um die Klustern bennennen zu können, haben wir mehrere Möglichkeiten.
# Wenn wir eine feste liste an Themen haben können wir versuchen zu finden 
# an welchen Kluster ist jedes Thema näher. 
# Um das simpler zu machen, wir suchen die Top 5 Wörter aus den Klustern und dann weisen wir die Themen manuell zu.

vect = CountVectorizer(stop_words=english_stop_words)

legende={l: [] for l in kmeans.labels_}
top_n=10
del kmeans, model

for i in set(legende.keys()):

    X_i = vect.fit_transform(df[df['labels']==i]['clean_text'])
    vocabulary = vect.get_feature_names_out()
    ind_i = np.argsort(X_i.toarray().sum(axis=0))[-top_n:]
    words_i = [vocabulary[a] for a in ind_i]
    manuell_i = input(f"Schreib ein Label für die Kategorie: {words_i}­\nEnter für default\n")
    legende[i]=manuell_i if manuell_i else words_i

#Wir können das visualisieren, in dem wir eine Projektionsfunktion 
# um die n-dimensionale Projektionen in einer 2D Grafik darzustellen
mapper = umap.UMAP().fit(embeddings)
df['umap_x']= mapper.embedding_[:,0]
df['umap_y']= mapper.embedding_[:,1]
df['cluster'] = df['labels'].apply(lambda x: str(legende[x]))

del mapper, embeddings

#Jetzt bauen wir die Grafik zusammen: 
# 1. DatenPunkte aus der mapper Projektion
# 2. Nach clustern färben 
# 3. Legende 
# (usw)
fig = px.scatter(df, x='umap_x', y='umap_y', color='cluster', hover_name="cluster", hover_data=["extract"])
fig.write_html('./cloud.html')