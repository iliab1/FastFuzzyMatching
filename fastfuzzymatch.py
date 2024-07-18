import itertools
import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import rapidfuzz
from tqdm import tqdm
import time
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FastFuzzyMatch:
    def __init__(
            self,
            clean: True,
            embedding_model: Optional[Any] = None,
            dimensionality_reduction_model: Optional[Any] = None,
            clustering_model: Optional[Any] = None,
            fuzzy_model: Optional[Any] = None,
            fuzzy_scorer: Optional[Any] = None,
    ):
        self.clean = clean

        if embedding_model is None:
            vectorizer_params = {
                'analyzer': 'char',
                'ngram_range': (1, 4)
            }
            embedding_model = TfidfVectorizer(**vectorizer_params)

        self.embedding_model = embedding_model

        if dimensionality_reduction_model is None:
            self.dimensionality_reduction_model = None
        else:
            self.dimensionality_reduction_model = dimensionality_reduction_model

        if clustering_model is None:
            neighbors_params = {
                'n_neighbors': 1,
                'metric': 'cosine',
                'n_jobs': -1
            }
            self.clustering_model = NearestNeighbors(**neighbors_params)
        else:
            self.clustering_model = clustering_model

        if fuzzy_model is None:
            self.fuzzy_model = rapidfuzz
        else:
            self.fuzzy_model = fuzzy_model

        if fuzzy_scorer is None:
            self.fuzzy_scorer = rapidfuzz.fuzz.token_sort_ratio
        else:
            self.fuzzy_scorer = fuzzy_scorer

    @staticmethod
    def clean_text(
            df,
            column_name
    ):
        logging.info(f'Cleaning text in column: {column_name}')
        df.drop_duplicates(subset=[column_name]).reset_index(drop=True)
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].str.replace(r'[^\w\s]', '', regex=True)
        df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
        df[column_name] = df[column_name].str.strip()
        return df

    def embedding(
            self,
            clean: pd.Series,
            dirty: pd.Series
    ) -> Tuple[np.array, np.array]:
        logging.info('Creating embeddings for clean and dirty data.')
        # Vectorize the clean and dirty data
        clean_vec = self.embedding_model.fit_transform(clean.values.astype('U'))
        dirty_vec = self.embedding_model.transform(dirty)  # dirty.values.astype('U')

        return clean_vec, dirty_vec

    def dimensionality_reduction(
            self,
            clean_vec: np.array,
            dirty_vec: np.array
    ) -> Tuple[np.array, np.array]:
        logging.info('Performing dimensionality reduction.')
        # Dimensionality reduction
        reduced_clean_vec = self.dimensionality_reduction_model.fit_transform(clean_vec)
        reduced_dirty_vec = self.dimensionality_reduction_model.transform(dirty_vec)
        return reduced_clean_vec, reduced_dirty_vec

    def clustering(
            self,
            clean_vec: np.array,
            dirty_vec: np.array
    ) -> Tuple[np.array, np.array]:
        logging.info('Clustering data.')
        # Fit the nearest neighbors
        self.clustering_model.fit(clean_vec)
        # Find the nearest neighbors
        distances, indices = self.clustering_model.kneighbors(dirty_vec, n_neighbors=1)

        return distances, indices

    def similarity_search(
            self,
            clean: pd.Series,
            dirty: pd.Series
    ) -> np.array:
        logging.info('Starting similarity search.')
        clean_vec, dirty_vec = self.embedding(clean, dirty)

        if self.dimensionality_reduction_model is not None:
            clean_vec, dirty_vec = self.dimensionality_reduction(clean_vec, dirty_vec)

        distances, indices = self.clustering(clean_vec, dirty_vec)
        nearest_values = np.array(clean)[indices]

        return nearest_values

    def fuzzy_match(
            self,
            row: str,
            match_candidates: List[str]
    ) -> List[Tuple[str, str, int]]:

        enumerated_matches = dict(enumerate(match_candidates))

        row_matches = self.fuzzy_model.process.extract(
            row, enumerated_matches, scorer=self.fuzzy_scorer, limit=5)

        result = [(row, match[0], match[1]) for match in row_matches]
        return result

    def fuzzy_search(
            self,
            clean: pd.Series,
            dirty: pd.Series,
            clean_column_name: str,
            dirty_column_name: str
    ) -> pd.DataFrame:

        # Find the nearest values
        nearest_values = self.similarity_search(clean, dirty)

        logging.info('Starting fuzzy search.')
        # For each row in the dirty data, find the best matches using its nearest values in the clean data
        results = []
        with logging_redirect_tqdm():
            for i, row in enumerate(tqdm(dirty, desc="Fuzzy Matching Progress")):
                results.append(self.fuzzy_match(row, nearest_values[i]))

        # Flatten the results and convert them to a DataFrame
        df = pd.DataFrame(
            itertools.chain.from_iterable(results),
            columns=[dirty_column_name, clean_column_name, 'Ratio']
        )
        return df

    def find_matches(
            self,
            clean_df: pd.DataFrame,
            clean_column: str,
            dirty_df: pd.DataFrame,
            dirty_column: str,
    ) -> pd.DataFrame:

        # Check if the columns exist
        if clean_column not in clean_df.columns:
            raise KeyError(f"'{clean_column}' not found in clean_df columns")
        if dirty_column not in dirty_df.columns:
            raise KeyError(f"'{dirty_column}' not found in dirty_df columns")

        # Pre-process data
        clean_df = clean_df.copy()
        dirty_df = dirty_df.copy()
        if self.clean:
            clean_df = self.clean_text(clean_df, clean_column)
            dirty_df = self.clean_text(dirty_df, dirty_column)

        # Take the subset directly to speed up the process
        # Find matches
        start = time.time()
        result = self.fuzzy_search(
            clean=clean_df[clean_column],
            dirty=dirty_df[dirty_column],
            clean_column_name=clean_column,
            dirty_column_name=dirty_column
        )
        end = time.time()
        logging.info('Fuzzy matching completed in {} seconds'.format(end - start))

        return result
