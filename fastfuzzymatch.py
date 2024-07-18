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
            clean: bool,
            merge: bool,
            embedding_model: Optional[Any] = None,
            dimensionality_reduction_model: Optional[Any] = None,
            clustering_model: Optional[Any] = None,
            fuzzy_model: Optional[Any] = None,
            fuzzy_scorer: Optional[Any] = None,
    ):
        self.clean = clean
        self.merge = merge

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
            df: pd.DataFrame,
            column_name: str
    ) -> pd.DataFrame:
        # Add original column to the DataFrame before cleaning
        original_column_name = f'original_{column_name}'
        df[original_column_name] = df[column_name]
        # Clean text data in the specified column of the DataFrame
        logging.info(f'Cleaning text in column: {column_name}')
        df.drop_duplicates(subset=[column_name], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].str.replace(r'[^\w\s]', '', regex=True)
        df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
        df[column_name] = df[column_name].str.strip()
        return df

    def embedding(
            self,
            clean: pd.Series,
            dirty: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Create embeddings for clean and dirty data
        logging.info('Creating embeddings for clean and dirty data.')
        clean_vec = self.embedding_model.fit_transform(clean.values.astype('U'))
        dirty_vec = self.embedding_model.transform(dirty.values.astype('U'))
        return clean_vec, dirty_vec

    def dimensionality_reduction(
            self,
            clean_vec: np.ndarray,
            dirty_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Perform dimensionality reduction
        logging.info('Performing dimensionality reduction.')
        reduced_clean_vec = self.dimensionality_reduction_model.fit_transform(clean_vec)
        reduced_dirty_vec = self.dimensionality_reduction_model.transform(dirty_vec)
        return reduced_clean_vec, reduced_dirty_vec

    def clustering(
            self,
            clean_vec: np.ndarray,
            dirty_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Perform clustering on the data
        logging.info('Clustering data.')
        self.clustering_model.fit(clean_vec)
        distances, indices = self.clustering_model.kneighbors(dirty_vec, n_neighbors=1)
        return distances, indices

    def similarity_search(
            self,
            clean: pd.Series,
            dirty: pd.Series
    ) -> np.ndarray:
        # Conduct similarity search using aforementioned methods
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
        # Perform fuzzy matching between a row and a list of match candidates
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
        # Perform fuzzy search on all rows in the dirty data
        nearest_values = self.similarity_search(clean, dirty)
        logging.info('Starting fuzzy search.')
        results = []
        with logging_redirect_tqdm():
            for i, row in enumerate(tqdm(dirty, desc="Fuzzy Matching Progress")):
                results.append(self.fuzzy_match(row, nearest_values[i]))

        df = pd.DataFrame(
            itertools.chain.from_iterable(results),
            columns=[dirty_column_name, 'Result', 'score']
        )
        return df

    def find_matches(
            self,
            clean_df: pd.DataFrame,
            clean_column: str,
            dirty_df: pd.DataFrame,
            dirty_column: str,
    ) -> pd.DataFrame:
        # Main function
        if clean_column not in clean_df.columns:
            raise KeyError(f"'{clean_column}' not found in clean_df columns")
        if dirty_column not in dirty_df.columns:
            raise KeyError(f"'{dirty_column}' not found in dirty_df columns")

        if clean_column == dirty_column:
            raise ValueError('clean_column and dirty_column cannot be the same')

        clean_df_copy = clean_df.copy()
        dirty_df_copy = dirty_df.copy()

        # Clean text if clean is True
        if self.clean:
            clean_df_copy = self.clean_text(clean_df_copy, clean_column)
            dirty_df_copy = self.clean_text(dirty_df_copy, dirty_column)

        start = time.time() # Set timer
        result = self.fuzzy_search(
            clean=clean_df_copy[clean_column],
            dirty=dirty_df_copy[dirty_column],
            clean_column_name=clean_column,
            dirty_column_name=dirty_column
        )
        end = time.time()

        logging.info('Fuzzy matching completed in {} seconds'.format(end - start))

        # Merge the result with the original data if merge is True
        if self.merge:
            # Merge the result with the original data
            logging.info("Initial merge with dirty dataframe")

            # Merge with the left (dirty) DataFrame
            merged_df = pd.merge(
                left=dirty_df_copy,
                right=result,
                how='left',
                left_on=dirty_column,
                right_on=dirty_column
            )
            logging.info("Merge result with clean dataframe")

            merged_df_right = pd.merge(
                merged_df,
                clean_df_copy,
                how='left',
                left_on='Result',
                right_on=clean_column
            )

            return merged_df_right
        else:
            return result
