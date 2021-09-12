from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


transformer = SentenceTransformer("paraphrase-mpnet-base-v2")
embeddings = pd.read_csv('new_embeddings_v2_dis.csv')
posts_df = pd.read_csv('combined_dis3.csv')

def update_url(url): 
  index = url.find('//t/') 
  if index == -1:
    return url 
  else: 
    return url[0:index] + url[index+1: len(url)]

def predict(post):
    recTitles = []
    recUrls = []

    #generating word embeding for post
    post_embedding = transformer.encode(post)

    #Getting Similarities
    similarities = cosine_similarity(post_embedding, embeddings)
    recs = similarities[0].argsort()[-5:][::-1]

    #Taking 5 most similar
    for recNum in recs:
      recTitles.append(posts_df['Topic Title'][recNum])
      recUrls.append(update_url(posts_df['url'][recNum]))

    return recTitles, recUrls
