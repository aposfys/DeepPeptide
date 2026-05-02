import pandas as pd
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset
from unittest.mock import patch, mock_open
import sys

# Mock read_csv to return a dataframe without 'coordinates'
mock_data = pd.DataFrame({
    'protein_id': ['A', 'B'],
    'propeptide_coordinates': ['(1-5)', '(10-15)'],
    'sequence': ['AAAAAAAAAA', 'BBBBBBBBBBBBBBBBBBBB'],
    'organism': ['human', 'human']
}).set_index('protein_id')

mock_partitioning = pd.DataFrame({
    'AC': ['A', 'B'],
    'cluster': [0, 0]
}).set_index('AC')

with patch('pandas.read_csv', side_effect=[mock_data, mock_partitioning]):
    try:
        ds = PrecomputedCSVForOverlapCRFDataset('data/embeddings', 'data.csv', 'part.csv')
        print("Success! Dataset loaded without 'coordinates' column.")
        print("Peptides parsed:", ds.peptides)
    except Exception as e:
        print("Failed:", e)
