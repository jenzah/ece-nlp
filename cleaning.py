import re
from typing import List, Optional

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

    def __init__(self, target: str = "genres", verbose: bool = True):
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

    def profile_unique_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Profiles columns to find categories, numeric ranges, or median list item counts."""
        profile_results = []

        for col in df.columns:
            if df[col].nunique(dropna=False) < 2:
                continue

            series_no_na = df[col].dropna()
            clean_unique = series_no_na.unique()

            row_data = {
                'column': col,
                'type': str(df[col].dtype),
                'unique_count': len(clean_unique),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                row_data['analysis_type'] = 'Numeric'
                row_data['detail'] = f"Min: {series_no_na.min()} | Max: {series_no_na.max()}"
                row_data['example'] = str(series_no_na.iloc[0]) if not series_no_na.empty else "NaN"

            elif col in self.LIST_LIKE_COLUMNS:
                row_data['analysis_type'] = 'List-like'
                item_counts = series_no_na.apply(
                    lambda x: len(str(x).split(',')) if str(x).lower() != 'unset' else 0
                )
                if not item_counts.empty:
                    median_len = int(item_counts.median())
                    row_data['detail'] = f"Median elements: {median_len} (Max: {int(item_counts.max())})"
                    median_match = item_counts[item_counts == median_len].index
                    row_data['example'] = (
                        f"Typical: {series_no_na.loc[median_match[0]]}" if len(median_match) else "N/A"
                    )
                else:
                    row_data['detail'] = "No valid list items"
                    row_data['example'] = "N/A"

            else:
                row_data['analysis_type'] = 'Categorical'
                row_data['detail'] = "Textual categories"
                row_data['example'] = str(clean_unique[0]) if len(clean_unique) > 0 else "NaN"

            profile_results.append(row_data)

        return pd.DataFrame(profile_results)

    def find_fuzzy_duplicates(self, df: pd.DataFrame,
                              threshold: int = 85,
                              min_length: int = 4) -> pd.DataFrame:
        """
        Finds groups of similar strings in textual columns.
        Ignores list-like columns to avoid noise.
        """
        issue_list = []
        potential_cols = [c for c in self._string_columns(df) if c not in self.LIST_LIKE_COLUMNS]

        for col in potential_cols:
            unique_values = [
                str(x) for x in df[col].dropna().unique()
                if len(str(x)) >= min_length
            ]
            if len(unique_values) < 2 or len(unique_values) > 1000:
                continue

            duplicate_groups = []
            processed_values = set()

            for i, val in enumerate(unique_values):
                if val in processed_values:
                    continue
                matches = process.extract(
                    val,
                    unique_values[i + 1:],
                    scorer=fuzz.ratio,
                    score_cutoff=threshold,
                )
                if matches:
                    group = [val] + [match[0] for match in matches]
                    duplicate_groups.append(sorted(group))
                    processed_values.update(group)

            if duplicate_groups:
                issue_list.append({
                    'column': col,
                    'found_groups': len(duplicate_groups),
                    'examples': duplicate_groups[:3],
                })

        return pd.DataFrame(issue_list)

    def analyse_mutual_information(self, df: pd.DataFrame,
                                   top_n: int = 20,
                                   training: bool = False) -> List[str]:
        """Calculates (and optionally plots) mutual information scores. Returns top feature names."""
        cols_to_skip = [self.target] + [c for c in self.LIST_LIKE_COLUMNS if c in df.columns]
        X = df.drop(columns=cols_to_skip)
        y = df[self.target]

        X_prepared = X.copy()
        discrete_mask = []
        for col in X_prepared.columns:
            if pd.api.types.is_object_dtype(X_prepared[col]) or pd.api.types.is_string_dtype(X_prepared[col]):
                X_prepared[col] = X_prepared[col].astype('category').cat.codes
                discrete_mask.append(True)
            else:  # numeric columns are assumed continuous
                X_prepared[col] = X_prepared[col].fillna(-999)
                discrete_mask.append(False)

        y_encoded = LabelEncoder().fit_transform(y)

        mi_scores = mutual_info_classif(
            X_prepared, y_encoded, discrete_features=discrete_mask, random_state=42
        )
        mi_df = (pd.DataFrame({'feature': X.columns, 'mutual_information': mi_scores})
                 .sort_values('mutual_information', ascending=False))

        top_features = mi_df.head(top_n)['feature'].tolist()

        if not training:
            self._vprint(f"\n--- Top {top_n} Features (Mutual Information) ---")
            self._vprint(mi_df.head(top_n).to_string(index=False))

            fig = px.bar(mi_df.head(top_n), x='mutual_information', y='feature', orientation='h',
                         title=f'Top {top_n} Features by Mutual Information',
                         labels={'mutual_information': 'Mutual Information Score', 'feature': 'Feature'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.show()

        return top_features

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

    def drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops identifiers, image paths and redundant fields."""
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        self._vprint(f"✓ Dropped {len(cols_to_drop)} useless columns: {cols_to_drop}")
        return df

    def drop_empty_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows where the target column is null, empty, or 'nan'."""
        if self.target not in df.columns:
            self._vprint(f"⚠️ Warning: '{self.target}' column not found. Skipping drop.")
            return df.copy()

        empty_mask = self._is_null_like(df[self.target])
        result = df[~empty_mask].copy()
        self._vprint(f"✓ Dropped {int(empty_mask.sum())} rows with empty '{self.target}'")
        return result

    def standardise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercases, strips whitespace / trailing semicolons, and unifies NaNs in string columns."""
        df = df.copy()
        string_cols = self._string_columns(df)

        for col in string_cols:
            df[col] = (df[col]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .str.replace(r';\s*$', '', regex=True)  # remove trailing semicolons
                       .str.strip()
                       .replace(self.NULL_STRINGS, np.nan))

        self._vprint(f"✓ Standardised {len(string_cols)} columns (lowercase, stripped, null-unified)")
        return df

    def drop_highly_empty_columns(self, df: pd.DataFrame,
                                  threshold: float = 0.8,
                                  empty_values: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Drops columns where more than `threshold` of values are empty/null.

        Args:
            threshold: Fraction of empty values above which a column is dropped
            empty_values: Additional string values to treat as empty (e.g. ['unset'])
        """
        extra_empty = [v.lower() for v in (empty_values or [])]
        total_rows = len(df)
        dropped_cols = []

        for col in df.columns:
            empty_mask = df[col].isna() | df[col].astype(str).str.lower().isin(extra_empty)
            empty_count = int(empty_mask.sum())
            empty_pct = empty_count / total_rows if total_rows else 0

            if empty_pct > threshold:
                dropped_cols.append(col)
                self._vprint(f"-> Dropping '{col}': {empty_pct:.1%} empty ({empty_count}/{total_rows})")

        self._vprint(f"\nDropped {len(dropped_cols)} highly empty columns in total.")
        return df.drop(columns=dropped_cols)

    def standardise_numeric(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts columns to numeric; dirty strings and -999 sentinels become NaN."""
        df = df.copy()
        cols_to_fix = [c for c in (columns or self.NUMERIC_COLS) if c in df.columns]

        for col in cols_to_fix:
            self._vprint(f"-> Converting to numeric: '{col}'")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == -999, col] = np.nan

        self._vprint(f"✓ Standardised {len(cols_to_fix)} numeric columns")
        return df

    def standardise_boolean(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts columns to bool. Unrecognised / null-like values become False."""
        df = df.copy()
        cols_to_fix = [c for c in (columns or self.BOOLEAN_COLS) if c in df.columns]

        bool_map = {
            'true': True, '1': True, '1.0': True, 't': True, 'yes': True,
            'false': False, '0': False, '0.0': False, 'f': False, 'no': False,
        }

        for col in cols_to_fix:
            self._vprint(f"-> Converting to boolean: '{col}'")
            df[col] = (df[col]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .map(bool_map)
                       .eq(True))  # NaN (unmapped) -> False

        self._vprint(f"✓ Standardised {len(cols_to_fix)} boolean columns")
        return df

    @staticmethod
    def _normalise_ordered_list(s, delimiter_pattern: str = r'[\+/\|;]') -> str:
        """Splits on delimiters, deduplicates while preserving order, rejoins with ', '."""
        if not isinstance(s, str) or not s.strip() or s.strip().lower() == 'nan':
            return 'unset'
        items = [item.strip() for item in re.split(delimiter_pattern, s) if item.strip()]
        result = ", ".join(dict.fromkeys(items))
        return result if result else 'unset'

    def convert_strings_to_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalises list-like columns (e.g. 'cast', 'directors') without sorting,
        so the original order is kept (lead actor stays first).
        """
        df = df.copy()
        cols = [c for c in self.LIST_LIKE_COLUMNS if c in df.columns]

        for col in cols:
            self._vprint(f"-> Normalising order-sensitive list: '{col}'")
            df[col] = df[col].apply(self._normalise_ordered_list)

        self._vprint(f"✓ Normalised {len(cols)} list columns (order preserved)")
        return df

    def expand_list_columns(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            n_items: int = 3) -> pd.DataFrame:
        """
        Splits list-like columns into n_items separate columns (e.g. cast_1, cast_2, cast_3)
        and drops the original column. The target is never expanded.
        """
        df = df.copy()
        candidates = columns or self.LIST_LIKE_COLUMNS
        cols_to_expand = [c for c in candidates if c != self.target and c in df.columns]

        for col in cols_to_expand:
            self._vprint(f"-> Expanding list-like column: '{col}'")
            expanded = df[col].apply(
                lambda x: str(x).split(', ') if str(x).lower() != 'unset' else []
            )
            for i in range(1, n_items + 1):
                df[f"{col}_{i}"] = expanded.apply(
                    lambda x, i=i: x[i - 1].strip() if len(x) >= i else np.nan
                )
            df = df.drop(columns=[col])

        self._vprint(f"✓ Expanded {len(cols_to_expand)} columns into {n_items} features each")
        return df

    def truncate_target(self, df: pd.DataFrame, max_items: int = 3) -> pd.DataFrame:
        """
        Keeps at most max_items entries in the target column
        (splits on commas and semicolons, rejoins with ', ').
        """
        if self.target not in df.columns:
            self._vprint(f"⚠️ Warning: '{self.target}' column not found. Skipping truncation.")
            return df.copy()

        def truncate_one(val):
            if pd.isna(val):
                return val
            items = [g.strip() for g in re.split(r'[,;]', str(val)) if g.strip()]
            return ', '.join(items[:max_items])

        df = df.copy()
        df[self.target] = df[self.target].apply(truncate_one)
        self._vprint(f"✓ '{self.target}' truncated to a maximum of {max_items} per row")
        return df

    # ------------------------------------------------------------------
    # CONVENIENCE PIPELINE
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the standard cleaning sequence and returns the cleaned DataFrame."""
        return (df
                .pipe(self.standardise_column_names)
                .pipe(self.drop_useless_columns)
                .pipe(self.standardise_data)
                .pipe(self.drop_empty_target)
                .pipe(self.truncate_target)
                .pipe(self.standardise_numeric)
                .pipe(self.standardise_boolean)
                .pipe(self.convert_strings_to_lists))


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