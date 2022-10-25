import pandas as pd
import numpy as np

import regex as re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_clean(dataset):
    """
    in:
    ---
    dataset: string, dataset name (supported: 'latest-small', '1m', '25m')

    out:
    ----

    """
    if dataset not in ['latest-small', '1m', '25m']:
        return
    
    if dataset=='1m':
        suffix='dat'
        delimiter='::'
        engine='python'
        encoding_errors='replace'
    else:
        suffix='csv'
        delimiter=','
        engine=None
        encoding_errors='strict'

    df = pd.read_csv(f'ml-{dataset}/movies.{suffix}', encoding_errors=encoding_errors, delimiter=delimiter, engine=engine, names=['movieId', 'title', 'genres'], header=0)
    df_rat = pd.read_csv(f'ml-{dataset}/ratings.{suffix}', encoding_errors=encoding_errors, delimiter=delimiter, engine=engine, names=['userId', 'movieId', 'rating', 'timestamp'], header=0)
    genome_scores = pd.read_csv(f'ml-{dataset}/genome-scores.{suffix}') if dataset=='25m' else None

    df['clean_title'] = df['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    dfs = (df, df_rat, genome_scores) if dataset=='25m' else (df, df_rat)
    return  dfs

def search(term, title_col, data):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(data[title_col])
    term = re.sub(r'[^a-zA-Z0-9 ]', '', term)
    term_vector = vectorizer.transform([term])
    similarity = cosine_similarity(term_vector, X).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = data.loc[indices][::-1].set_index(np.arange(1,6))
    return results


def recommend(movie_id, df_rat, data, n=10, threshold=.1):
    # Establish similarity condition and filter userId column
    condition = (df_rat['movieId']==movie_id) & (df_rat['rating']>4)
    sim_users = df_rat.loc[condition, 'userId'].unique()
    
    # Establish recommendation condition
    rec_condition = (df_rat['userId'].isin(sim_users))&(df_rat['rating']>4)
    counts = df_rat.loc[rec_condition, 'movieId'].value_counts() / len(sim_users)
    # Get percentages of users liking recommended movies
    recs_sim_perc = counts[counts>threshold]
    recs_movie_ids = recs_sim_perc.index
    
    # Establish how these movies are recommended by general public
    base_condition = (df_rat['movieId'].isin(recs_movie_ids))&(df_rat['rating']>4)
    all_users = df_rat.loc[base_condition, 'userId'].unique()
    recs_all_perc = df_rat.loc[base_condition, 'movieId'].value_counts() / len(all_users)
    
    # Compute recommendation score
    score_df = pd.concat([recs_sim_perc, recs_all_perc], axis=1)
    score_df.columns = ['Similar', 'All']
    score_df['Score'] = score_df['Similar'] / score_df['All']

    top = (score_df.sort_values('Score',ascending=False).iloc[:n]
             .merge(data[['movieId', 'title', 'genres']], left_index=True, right_on='movieId')
             .drop(['movieId', 'Similar', 'All'], axis=1)
             )
    return top


def tag_recommender(movie_id, df_tags, n=10):
    """Takes movie id, df (with tag ids and relevances) and returns np.array with n movieIds of similarly tagged pictures"""
    if movie_id not in df_tags['movieId'].values:
        return None
    else:
        # orig_tags filters original data to entries with a given movieId and high tag relevance scores
        orig_tags = df_tags[(df_tags['movieId']==movie_id)&(df_tags['relevance']>.8)]
        
        same_tags_cond = (df_tags['tagId'].isin(orig_tags['tagId']))&(df_tags['relevance']>.8)
        sim_tagged_movies = df_tags[same_tags_cond].copy()
        
        # Let's keep track of how many tags movies have in common with the given one
        same_tags_count_mapping = sim_tagged_movies['movieId'].value_counts()
        sim_tagged_movies['same_tags_count'] = sim_tagged_movies['movieId'].map(same_tags_count_mapping)
        
        # Returns np.array of unique movie ids based on how many tags movie have in common with the given one
        return data[data['movieId'].isin(sim_tagged_movies.sort_values(['same_tags_count', 'movieId'], ascending=False)['movieId'].unique()[:n])]