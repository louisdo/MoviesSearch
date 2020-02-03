# MoviesSearch
A simple search engine using term frequency-inverse document frequenzy (TFIDF)<br />
I saw my cousin making this, got curious and tried to make my own

# Prepare data
  ```
  rem "download general data from:  https://drive.google.com/file/d/1CZsJGWS9hZ7z2t_fJcmxnn-fVo_EtM5P/view?usp=sharing"
  rem "create a folder named 'data'"
  rem "move the csv file just downloaded in to 'data'"
  
  python prepare_data.create_tokenize_data.py --csv-in ./data/general_movies_data.csv --csv-out ./data/tokenized_data.csv
  ```

# How to run
```python
>>> from MoviesSearch import MoviesSearchEngine
>>> search_engine=MoviesSearchEngine("path/to/tokenized/data","path/to/general/data")
>>> search_engine.search("woody and buzz lightyear")
[(1, 'Toy Story'), (2, 'Toy Story 3'), (3, 'Toy Story 2'), (4, 'In the Shadow of the Moon'), (5, 'For Your Consideration')]
```
