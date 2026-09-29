import re
from typing import List, Optional
import html
import unicodedata

import numpy as np
import pandas as pd
#mport plotly.express as px
#rom rapidfuzz import process, fuzz
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

#Pour fonction remove_numbers
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
#pip install nltk

class DataCleaner:
    """
    The cleaner holds only configuration (target name, verbosity), never data.
    - Transform methods take a DataFrame and return a NEW DataFrame
      (the input is never modified in place).
    - Analysis methods (audit / profile / fuzzy / mutual information)
      take a DataFrame and return a report, leaving the data untouched.
    """

    NULL_STRINGS = ['nan', 'none', '']

    def __init__(self, target: str = "fake_news", verbose: bool = True):
        self.target = target
        self.verbose = verbose

    def _vprint(self, *args, **kwargs):
        """Internal helper to print only if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    @staticmethod
    def _string_columns(df: pd.DataFrame) -> pd.Index:
        return df.select_dtypes(include=['object', 'string']).columns

    @staticmethod
    def _is_null_like(series: pd.Series) -> pd.Series:
        """Boolean mask: True for NaN, 'nan' and empty strings."""
        clean_s = series.astype(str).str.strip().str.lower()
        return series.isna() | (clean_s == 'nan') | (clean_s == '')

    @staticmethod
    def merge_fake_true(fake: pd.DataFrame,
                        true: pd.DataFrame,
                        label_col: str = "fake_news",
                        fake_value: int = 1,
                        true_value: int = 0,
                        shuffle: bool = True,
                        drop_duplicates: bool = False,
                        random_state: int = 42,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Merges the fake and true news datasets into one DataFrame,
        adding a label column that flags each row's origin.
    
        Args:
            fake: DataFrame of fake news
            true: DataFrame of true news
            label_col: Name of the added flag column
            fake_value: Label value given to fake articles (default 1)
            true_value: Label value given to true articles (default 0)
            shuffle: Shuffle rows so fake and true articles are interleaved
            drop_duplicates: Drop rows that are identical across all original columns
            random_state: Seed for reproducible shuffling
            verbose: Print a summary of the merge
    
        Returns:
            Merged DataFrame with a fresh index and the label column
        """
        if label_col in fake.columns or label_col in true.columns:
            raise ValueError(f"Column '{label_col}' already exists in one of the datasets.")
    
        only_fake = set(fake.columns) - set(true.columns)
        only_true = set(true.columns) - set(fake.columns)
        if (only_fake or only_true) and verbose:
            print(f"⚠️ Column mismatch — only in fake: {sorted(only_fake)}, "
                  f"only in true: {sorted(only_true)} (missing values filled with NaN)")
    
        merged = pd.concat(
            [fake.assign(**{label_col: fake_value}),
             true.assign(**{label_col: true_value})],
            ignore_index=True,
        )
    
        if drop_duplicates:
            content_cols = [c for c in merged.columns if c != label_col]
            before = len(merged)
            merged = merged.drop_duplicates(subset=content_cols)
            if verbose:
                print(f"✓ Dropped {before - len(merged)} duplicate rows")
    
        if shuffle:
            merged = merged.sample(frac=1, random_state=random_state)
    
        merged = merged.reset_index(drop=True)
    
        if verbose:
            counts = merged[label_col].value_counts()
            print(f"✓ Merged {len(merged)} rows — "
                  f"fake: {counts.get(fake_value, 0)}, true: {counts.get(true_value, 0)}")
    
        return merged

    # ------------------------------------------------------------------
    # ANALYSIS (return reports, never modify the data)
    # ------------------------------------------------------------------

    def audit_column_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a summary of column health, catching all null variants."""
        audit_data = []
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                nan_count = self._is_null_like(series).sum()
            else:
                nan_count = series.isna().sum()

            audit_data.append({
                'column': col,
                'type': str(series.dtype),
                'missing_or_nan': nan_count,
                'total_rows': len(series),
            })
        return pd.DataFrame(audit_data)

    # ------------------------------------------------------------------
    # TRANSFORMS (take a df, return a new df)
    # ------------------------------------------------------------------

    def standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes control characters and extra spaces from column names.
        Logs every change for audit purposes.
        """
        df = df.copy()
        self._vprint("\n" + "=" * 80)
        self._vprint("-> STARTING COLUMN NAME STANDARDISATION")
        self._vprint("=" * 80)

        new_cols = []
        changes = []
        for col in df.columns:
            cleaned = "".join(char if char.isprintable() else ' ' for char in str(col))
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            new_cols.append(cleaned)
            if cleaned != col:
                changes.append({'original': repr(col), 'standardised': cleaned})

        df.columns = new_cols

        if not changes:
            self._vprint("✓ All column names were already standard. No changes made.")
        else:
            self._vprint(f"⚠️ Found and corrected {len(changes)} non-standard column names:")
            self._vprint(f"\n{'ORIGINAL (RAW)':<50} | {'STANDARDISED':<30}")
            self._vprint("-" * 80)
            for change in changes:
                self._vprint(f"{change['original']:<50} | {change['standardised']:<30}")

        self._vprint("=" * 80 + "\n")
        return df

    def standardise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercases, strips whitespace / trailing semicolons, collapses inner whitespace and unifies NaNs."""
        df = df.copy()
        string_cols = self._string_columns(df)

        for col in string_cols:
            df[col] = (df[col]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .str.replace(r';\s*$', '', regex=True)  # remove trailing semicolons
                       .str.replace(r'\s+', ' ', regex=True)   # collapse tabs, newlines, repeated spaces
                       .str.strip()
                       .replace(self.NULL_STRINGS, np.nan))

        self._vprint(f"✓ Standardised {len(string_cols)} columns (lowercase, stripped, whitespace collapsed, null-unified)")
        return df

    def normalise_text(self, text, strip_accents: bool = True, keep_apostrophes: bool = False):
        """Removes HTML, escape sequences and punctuation from a single string."""
        if not isinstance(text, str):
            return text

        text = re.sub(r'<[^>]+>', ' ', text)           # HTML tags: <br>, <p class="x">, </a>
        text = re.sub(r'&#?\w+;', ' ', text)           # HTML entities: &amp;, &nbsp;, &#39;
        text = re.sub(r'\\\w+', ' ', text)             # escape sequences: \u2013, \xa0, \n

        if strip_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))

        if keep_apostrophes:
            text = re.sub(r"[^\w\s'’]", ' ', text)
            text = re.sub(r"(?<!\w)['’]|['’](?!\w)", ' ', text)  # keep ' only inside words (don't)
        else:
            text = re.sub(r"['’]", '', text)                      # don't -> dont (no split)
            text = re.sub(r'[^\w\s]', ' ', text)

        text = text.replace('_', ' ')                  # \w keeps underscores, drop them too
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else np.nan

    # ------------------------------------------------------------------
    # CONVENIENCE PIPELINE
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the standard cleaning sequence and returns the cleaned DataFrame."""
        return (df
                .pipe(self.standardise_column_names)
                .pipe(self.standardise_data)
                .pipe(self.normalise_text)
                )


    def remove_numbers(self, df: pd.DataFrame,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Supprime les nombres des colonnes textuelles.

        Args:
            df: DataFrame à nettoyer.
            columns: Colonnes sur lesquelles appliquer la suppression.
                    Par défaut : title et text.

        Returns:
            Nouveau DataFrame sans les nombres.
        """

        df = df.copy()

        columns = columns or ["title", "text"]

        for col in columns:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.replace(r"\d+", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

        self._vprint(f"✓ Numbers removed from: {columns}")

        return df


    def remove_stopwords(self, df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        language: str = "english") -> pd.DataFrame:
        """
        Supprime les mots vides (stopwords) des colonnes textuelles.

        Args:
            df: DataFrame à nettoyer.
            columns: Colonnes sur lesquelles appliquer la suppression.
            language: Langue des stopwords ('english', 'french', etc.).

        Returns:
            Nouveau DataFrame sans les stopwords.
        """

        df = df.copy()

        columns = columns or ["title", "text"]

        stop_words = set(stopwords.words(language))

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda text: " ".join(
                        word for word in str(text).split()
                        if word.lower() not in stop_words
                    )
                )

        self._vprint(
            f"✓ Stopwords removed from: {columns} "
            f"(language: {language})"
        )

        return df

    def remove_duplicate_words(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Supprime les mots dupliqués dans les colonnes textuelles.
        Une seule occurrence de chaque mot est conservée.
        """

        df = df.copy()

        columns = columns or ["title", "text"]

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda text: " ".join(
                        dict.fromkeys(str(text).split())
                    )
                )

        self._vprint(f"✓ Duplicate words removed from: {columns}")

        return df

    def remove_duplicate_words(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Supprime les mots dupliqués dans les colonnes textuelles.
        Une seule occurrence de chaque mot est conservée.
        """

        df = df.copy()

        columns = columns or ["title", "text"]

        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda text: " ".join(
                        dict.fromkeys(str(text).split())
                    )
                )

        self._vprint(f"✓ Duplicate words removed from: {columns}")

        return df