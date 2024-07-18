# FastFuzzyMatch

`FastFuzzyMatch` is a Python class designed to perform efficient fuzzy matching between two datasets. 

It leverages a combination of text embedding, dimensionality reduction, clustering, and fuzzy matching techniques to find the closest matches between entries in a clean dataset and a dirty dataset.

## Features

- **Text Cleaning**: Preprocess text data to ensure consistency.
- **Text Embedding**: Transform text data into numerical vectors using `TfidfVectorizer`.
- **Dimensionality Reduction**: Optionally reduce the dimensionality of the vectors for faster computation.
- **Clustering**: Use `NearestNeighbors` for fast nearest-neighbor search.
- **Fuzzy Matching**: Perform fuzzy matching using `rapidfuzz` to find the most similar entries.
- **Result Merging**: Optionally merge the matching results with the original datasets.

## Installation

To use `FastFuzzyMatch`, you need to install the required Python packages:

```bash
pip install pandas numpy scikit-learn rapidfuzz tqdm
```

## Example
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import rapidfuzz
from fastfuzzymatch import FastFuzzyMatch

# Example data
clean_data = {
    'Name': ['Alpha Corporation', 'Beta Technologies', 'Gamma Enterprises'],
    'CusID': [1001, 1002, 1003]
}

dirty_data = {
    'name': ['Alpha Corp', 'Beta Tech', 'Gama Enterprizes'],
    'CaseID': [2001, 2002, 2003]
}

clean_df = pd.DataFrame(clean_data)
dirty_df = pd.DataFrame(dirty_data)

# Initialise FastFuzzyMatch
ffm = FastFuzzyMatch(
    clean=True,
    merge=True,
    embedding_model=TfidfVectorizer(analyzer='char', ngram_range=(1, 4)),
    clustering_model=NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1),
    fuzzy_model=rapidfuzz,
    fuzzy_scorer=rapidfuzz.fuzz.token_sort_ratio
)

# Run find_matches method
result = ffm.find_matches(
    clean_df=clean_df,
    clean_column='Name',
    dirty_df=dirty_df,
    dirty_column='name'
)

# Display results
print(result)

```
Result:
```
               name  CaseID             Result      score               Name
0        alpha corp    2001  alpha corporation  74.074074  alpha corporation   
1         beta tech    2002  beta technologies  69.230769  beta technologies   
2  gama enterprizes    2003  gamma enterprises  90.909091  gamma enterprises   
   CusID  
0   1001  
1   1002  
2   1003