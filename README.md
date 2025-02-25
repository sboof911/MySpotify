# MySpotify

MySpotify is a Python-based data processing and recommendation system for music data. It provides tools to preprocess large, chunked datasets, convert files to parquet format, merge multiple data sources, and generate various recommendations. The project includes modules for data preprocessing, collections management (with various retrieval methods, `baseline` approaches, `Word2Vec`, and an `MLPClassifier`), and recommendation generation using `AlternatingLeastSquares`.

## Project Structure

- **MySpotify.ipynb**  
  A Jupyter Notebook for interactive usage, testing, and demonstrations of the MySpotify functionalities.

  - [`MySpotify notebook`](MySpotify.ipynb)

- **MySpotify.py**  
  The main module that initializes the MySpotify class. It handles data unzipping and conversion, delegating tasks to other modules in the project. See [`MySpotify`](MySpotify.py) for details.

- **Collections/**  
   Contains modules related to managing various collections (love, war, happiness) of music data. This module includes functionality for various collection retrieval methods, baseline approaches, Word2Vec based retrieval, and an MLPClassifier for theme classification.

  - [`Collections.py`](Collections/Collections.py)
  - [`Classifier.py`](Collections/Classifier.py)

- **PrepocessData/**  
  Provides tools for file extraction and conversion:  
  - [`ReadZip.py`](PrepocessData/ReadZip.py) - Handles unzipping of data archives.  
  - [`ConvertFiles.py`](PrepocessData/ConvertFiles.py) - Converts datasets to parquet format.  
  - [`MergeData.py`](PrepocessData/MergeData.py) - Merges various datasets.  
  - [`utils.py`](PrepocessData/utils.py) - Utility functions for data preprocessing.

- **Recommendations/**  
    Module to generate music track recommendations. This module utilizes the AlternatingLeastSquares algorithm from `implicit.als` to classify and recommend music tracks.

    - [`Recommendations.py`](Recommendations/Recommendations.py).

- **TopTracks/**  
  Contains functions to retrieve the top listened tracks and top tracks by genre.

## Libraries Used

The project leverages several Python libraries including:

- **pyarrow** – For handling parquet file conversions.
- **pandas** – For data manipulation and analysis.
- **scikit-learn** – For implementing classification algorithms in the collections module.
- **implicit** - For Recommendation classification
