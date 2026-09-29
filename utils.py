import html
import os
import re
import unicodedata
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def _load_stopwords(language: str = "english") -> set:
    """Returns the NLTK stopword list, downloading it only if it is missing."""
    try:
        words = stopwords.words(language)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words(language)
    # Apostrophes are removed during normalisation ("don't" -> "dont"),
    # so the apostrophe-free variants must be stopwords too.
    return set(words) | {w.replace("'", "") for w in words}


class DataCleaner:
    """
    The cleaner holds only configuration (verbosity), never data.
    - Transform methods take a DataFrame and return a NEW DataFrame
      (the input is never modified in place, the index is preserved).
    - Analysis methods take a DataFrame and return a report,
      leaving the data untouched.

    Recommended order of the transforms:
        standardise_column_names -> standardise_data -> remove_source_artifacts
        -> normalise_text -> remove_numbers -> remove_stopwords
        -> drop_empty_and_duplicates -> add_content_column
    """

    NULL_STRINGS = ['nan', 'none', '']
    TEXT_COLUMNS = ["title", "text"]

    # Publisher / platform markers that reveal the class without describing the news itself.
    # Each entry can be commented out individually to measure its influence.
    SOURCE_PATTERNS = {
        "reuters_dateline": r"^[^\n]{0,150}?\(\s*reuters\s*\)\s*[-\u2013\u2014]\s*",  # "WASHINGTON (Reuters) - "
        "reuters_word": r"\breuters\b",
        "featured_image": r"\bfeatured\s+(?:image|photo)s?\b[^.\n]{0,120}",   # "Featured image via Getty Images"
        "image_credit": r"\b(?:image|photo)s?\s+(?:via|by|credit|courtesy)\b[^.\n]{0,120}",
        "getty_images": r"\bgetty\s+images\b",
        "read_more": r"\bread\s+more\s*:?",
        "twitter_pictures": r"\bpic\.twitter\.com/\S+",
        "twitter_handles": r"@\w+",
        "embed_credit": r"\bvia\s*:?\s*(?:youtube|twitter|facebook|instagram|vidme|www\S*)\b",  # "via YouTube"
        "twenty_first_century_wire": r"\b21st\s+century\s+wire\b(?:\s+says)?",
        "screen_capture": r"\bscreen\s*(?:capture|shot)s?\b",
        "title_tags": r"[\[\(]\s*(?:video|videos|tweet|tweets|image|images|details|watch|breaking|screenshots?)\s*[\]\)]",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _vprint(self, *args, **kwargs):
        """Internal helper to print only if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    @staticmethod
    def _string_columns(df: pd.DataFrame) -> pd.Index:
        return df.select_dtypes(include=['object', 'string']).columns

    @classmethod
    def _is_null_like(cls, series: pd.Series) -> pd.Series:
        """Boolean mask: True for NaN and for every string listed in NULL_STRINGS."""
        clean_s = series.astype(str).str.strip().str.lower()
        return series.isna() | clean_s.isin(cls.NULL_STRINGS)

    @staticmethod
    def _apply_to_text(df: pd.DataFrame, columns: List[str], func: Callable) -> pd.DataFrame:
        """Applies func to the string cells of the given columns; NaN cells are left untouched."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].map(lambda x: func(x) if isinstance(x, str) else x)
        return df

    @staticmethod
    def _empty_to_nan(text: str):
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else np.nan

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    def merge_fake_true(self,
                        fake: pd.DataFrame,
                        true: pd.DataFrame,
                        label_col: str = "fake_news",
                        fake_value: int = 1,
                        true_value: int = 0,
                        shuffle: bool = True,
                        random_state: int = 42,
                        output_path: Optional[str] = None) -> pd.DataFrame:
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
            random_state: Seed for reproducible shuffling
            output_path: If given, the merged DataFrame is saved there as CSV

        Returns:
            Merged DataFrame with a fresh index and the label column
        """
        if label_col in fake.columns or label_col in true.columns:
            raise ValueError(f"La colonne '{label_col}' existe déjà dans l'un des datasets.")

        only_fake = set(fake.columns) - set(true.columns)
        only_true = set(true.columns) - set(fake.columns)
        if only_fake or only_true:
            self._vprint(f"⚠️ Colonnes différentes — seulement dans fake : {sorted(only_fake)}, "
                         f"seulement dans true : {sorted(only_true)} (valeurs manquantes remplies par NaN)")

        merged = pd.concat(
            [fake.assign(**{label_col: fake_value}),
             true.assign(**{label_col: true_value})],
            ignore_index=True,
        )

        if shuffle:
            merged = merged.sample(frac=1, random_state=random_state)

        merged = merged.reset_index(drop=True)

        counts = merged[label_col].value_counts()
        self._vprint(f"✓ Fusion : {len(merged)} lignes — "
                     f"fake : {counts.get(fake_value, 0)}, true : {counts.get(true_value, 0)}")

        if output_path:
            folder = os.path.dirname(output_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            merged.to_csv(output_path, index=False)
            self._vprint(f"✓ Fichier fusionné sauvegardé : {output_path}")

        return merged

    # ------------------------------------------------------------------
    # TRANSFORMS (take a df, return a new df)
    # ------------------------------------------------------------------

    def standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes control characters and extra spaces from column names.
        Logs every change for audit purposes.
        """
        df = df.copy()
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
            self._vprint("✓ Noms de colonnes déjà standards, aucune modification.")
        else:
            self._vprint(f"⚠️ {len(changes)} noms de colonnes corrigés :")
            self._vprint(f"\n{'ORIGINAL (BRUT)':<50} | {'STANDARDISÉ':<30}")
            self._vprint("-" * 80)
            for change in changes:
                self._vprint(f"{change['original']:<50} | {change['standardised']:<30}")
        return df

    def standardise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercases, strips whitespace / trailing semicolons, collapses inner whitespace and unifies NaNs."""
        df = df.copy()
        string_cols = self._string_columns(df)

        for col in string_cols:
            is_null = self._is_null_like(df[col])
            df[col] = (df[col]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .str.replace(r';\s*$', '', regex=True)  # remove trailing semicolons
                       .str.replace(r'\s+', ' ', regex=True)   # collapse tabs, newlines, repeated spaces
                       .str.strip()
                       .mask(is_null))

        self._vprint(f"✓ {len(string_cols)} colonnes standardisées "
                     f"(minuscules, espaces nettoyés, valeurs nulles unifiées)")
        return df

    def remove_source_artifacts(self, df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                patterns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Removes publisher / platform markers (Reuters dateline, "featured image via",
        twitter embeds, [VIDEO] tags...) that give away the class without being news content.
        Must run BEFORE normalise_text, because the patterns rely on punctuation.
        """
        columns = columns or self.TEXT_COLUMNS
        patterns = patterns if patterns is not None else self.SOURCE_PATTERNS
        compiled = {name: re.compile(p, flags=re.IGNORECASE) for name, p in patterns.items()}

        hits = {name: 0 for name in compiled}

        def strip_markers(text: str):
            for name, regex in compiled.items():
                text, n = regex.subn(" ", text)
                hits[name] += n
            return self._empty_to_nan(text)

        df = self._apply_to_text(df, columns, strip_markers)

        self._vprint(f"✓ Marqueurs de source supprimés de : {columns}")
        for name, n in hits.items():
            self._vprint(f"   - {name:<28} {n:>7} occurrences")
        return df

    @staticmethod
    def _normalise_string(text: str, strip_accents: bool = True):
        """Decodes HTML, removes URLs / emails / tags / escape sequences and punctuation from a single string."""
        text = html.unescape(text)                                     # &amp; -> &, &#39; -> '
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)                           # HTML tags: <br>, <p class="x">
        text = re.sub(r'https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|gov)/\S*', ' ', text)  # URLs
        text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', ' ', text)          # e-mail addresses
        text = re.sub(r'\\\w+', ' ', text)                             # literal escape sequences: \u2013, \n

        if strip_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))

        text = re.sub(r'\bu\.s\.(?:a\.)?', ' usa ', text)              # "u.s." would become "us" (a stopword)
        text = re.sub(r'\b(?:[a-z]\.){2,}',                            # other acronyms: u.k. -> uk, d.c. -> dc
                      lambda m: m.group().replace('.', ''), text)
        text = re.sub(r"['’‘`´]s\b", '', text)                        # possessive: trump's -> trump
        text = re.sub(r"['’‘`´]", '', text)                           # don't -> dont (no split)
        text = re.sub(r'[^\w\s]', ' ', text)                           # remaining punctuation
        text = text.replace('_', ' ')                                  # \w keeps underscores, drop them too
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else np.nan

    def normalise_text(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       strip_accents: bool = True) -> pd.DataFrame:
        """Applies the single-string normalisation to every text column."""
        columns = columns or self.TEXT_COLUMNS
        df = self._apply_to_text(df, columns,
                                 lambda t: self._normalise_string(t, strip_accents=strip_accents))
        self._vprint(f"✓ Texte normalisé dans : {columns} "
                     f"(HTML décodé, URLs, e-mails et ponctuation supprimés)")
        return df

    def remove_numbers(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes whole tokens that start with a digit ("2017", "21st", "9th", "1990s"),
        so that no junk fragments such as "st" or "th" are left behind.
        """
        columns = columns or self.TEXT_COLUMNS
        df = self._apply_to_text(df, columns,
                                 lambda t: self._empty_to_nan(re.sub(r"\b\d+\w*\b", " ", t)))
        self._vprint(f"✓ Nombres supprimés de : {columns}")
        return df

    def remove_stopwords(self, df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         language: str = "english",
                         min_word_length: int = 2) -> pd.DataFrame:
        """
        Removes stopwords (NLTK list + apostrophe-free variants) and words
        shorter than min_word_length (leftover single letters such as "s" or "t").
        """
        columns = columns or self.TEXT_COLUMNS
        stop_words = _load_stopwords(language)

        def filter_words(text: str):
            kept = [w for w in text.split()
                    if w not in stop_words and len(w) >= min_word_length]
            return " ".join(kept) if kept else np.nan

        df = self._apply_to_text(df, columns, filter_words)
        self._vprint(f"✓ Stopwords supprimés de : {columns} "
                     f"(langue : {language}, {len(stop_words)} mots, longueur min : {min_word_length})")
        return df

    def drop_empty_and_duplicates(self, df: pd.DataFrame,
                                  text_col: str = "text",
                                  label_col: str = "fake_news") -> pd.DataFrame:
        """
        Drops rows whose text is missing or empty, then rows whose text is an exact
        duplicate of an earlier row (the first occurrence is kept).
        Run it after the text cleaning so near-duplicates that became identical are caught too.
        """
        before = len(df)
        empty_mask = self._is_null_like(df[text_col])
        df = df[~empty_mask]
        after_empty = len(df)

        # Texts present in both classes (same article labelled FAKE and REAL)
        conflicts = (df.groupby(text_col)[label_col].nunique() > 1).sum() if label_col in df.columns else 0

        df = df.drop_duplicates(subset=[text_col]).copy()

        self._vprint(f"✓ Articles vides supprimés : {before - after_empty}")
        self._vprint(f"✓ Doublons de '{text_col}' supprimés : {after_empty - len(df)}")
        if conflicts:
            self._vprint(f"⚠️ {conflicts} textes apparaissaient dans les deux classes (seule la 1re occurrence est gardée)")
        self._vprint(f"✓ Articles restants : {len(df)} (sur {before})")
        return df

    def add_content_column(self, df: pd.DataFrame,
                           title_col: str = "title",
                           text_col: str = "text",
                           new_col: str = "content") -> pd.DataFrame:
        """Concatenates title and body into a single text column used by the models."""
        df = df.copy()
        df[new_col] = (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.strip()
        self._vprint(f"✓ Colonne '{new_col}' créée ({title_col} + {text_col})")
        return df


class Explorer:
    """
    Exploration helpers: every method takes data and returns a table and/or draws a plot.
    Nothing is modified in place.
    """

    def __init__(self, target: str = "fake_news", label_names: Optional[Dict[int, str]] = None):
        self.target = target
        self.label_names = label_names or LABEL_NAMES

    def _class_texts(self, df: pd.DataFrame, text_col: str) -> Dict[str, pd.Series]:
        return {name: df.loc[df[self.target] == value, text_col].dropna()
                for value, name in self.label_names.items()}

    # ------------------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------------------

    @staticmethod
    def duplicates_report(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Counts empty values and duplicates for the given text columns, plus full-row duplicates."""
        columns = columns or ["title", "text"]
        rows = [{"vérification": "lignes entièrement dupliquées", "nombre": int(df.duplicated().sum())}]
        for col in columns:
            empty = df[col].isna() | df[col].astype(str).str.strip().eq("")
            rows.append({"vérification": f"{col} vides", "nombre": int(empty.sum())})
            rows.append({"vérification": f"{col} dupliqués", "nombre": int(df.duplicated(subset=[col]).sum())})
        return pd.DataFrame(rows).set_index("vérification")

    @staticmethod
    def before_after(before: pd.DataFrame, after: pd.DataFrame,
                     column: str = "text", n: int = 3, max_chars: int = 500) -> None:
        """Prints a few texts before and after cleaning (rows matched on the index)."""
        common = after.index.intersection(before.index)[:n]
        for idx in common:
            print("=" * 100)
            print(f"AVANT (ligne {idx}) :")
            print(str(before.at[idx, column])[:max_chars])
            print("\nAPRÈS :")
            print(str(after.at[idx, column])[:max_chars])
            print()

    # ------------------------------------------------------------------
    # CLASSES AND METADATA
    # ------------------------------------------------------------------

    def class_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Table and bar chart of the number of articles per class."""
        counts = df[self.target].value_counts().sort_index()
        table = pd.DataFrame({
            "Nombre": counts,
            "Pourcentage": (counts / counts.sum() * 100).round(2),
        })
        table.index = [self.label_names[i] for i in table.index]

        ax = table["Nombre"].plot(kind="bar", color=["tab:blue", "tab:red"], figsize=(6, 4))
        ax.set_title("Distribution des classes REAL / FAKE")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Nombre d'articles")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        return table

    def subject_by_class(self, df: pd.DataFrame, subject_col: str = "subject") -> pd.DataFrame:
        """Crosstab and bar chart of the subjects per class (leakage check)."""
        table = pd.crosstab(df[subject_col], df[self.target])
        table.columns = [self.label_names[c] for c in table.columns]
        table = table.sort_values(list(table.columns), ascending=False)

        table.plot(kind="bar", figsize=(10, 4), color=["tab:blue", "tab:red"])
        plt.title("Répartition des sujets selon la classe")
        plt.xlabel("Sujet")
        plt.ylabel("Nombre d'articles")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return table

    # ------------------------------------------------------------------
    # LENGTHS
    # ------------------------------------------------------------------

    def length_stats(self, df: pd.DataFrame,
                     text_col: str = "content", title_col: str = "title") -> pd.DataFrame:
        """Median, mean and extremes of the number of words per class."""
        lengths = pd.DataFrame({
            "mots_article": df[text_col].fillna("").str.split().str.len(),
            "mots_titre": df[title_col].fillna("").str.split().str.len(),
            "classe": df[self.target].map(self.label_names),
        })
        return (lengths.groupby("classe")[["mots_article", "mots_titre"]]
                .agg(["median", "mean", "min", "max"])
                .round(1))

    def plot_length_distribution(self, df: pd.DataFrame,
                                 text_col: str = "content", quantile: float = 0.99) -> None:
        """Overlaid histograms of the number of words; the x axis stops at the given quantile."""
        word_counts = df[text_col].fillna("").str.split().str.len()
        plt.figure(figsize=(8, 4))
        for value, name in self.label_names.items():
            plt.hist(word_counts[df[self.target] == value], bins=60, alpha=0.5, label=name)
        plt.xlim(0, word_counts.quantile(quantile))
        plt.xlabel("Nombre de mots")
        plt.ylabel("Nombre d'articles")
        plt.title("Distribution de la longueur des articles")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # VOCABULARY
    # ------------------------------------------------------------------

    @staticmethod
    def top_ngrams(texts: pd.Series, n: int = 20, ngram_range: tuple = (1, 1)) -> pd.DataFrame:
        """Most frequent n-grams (stopwords are expected to be already removed)."""
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=2)
        matrix = vectorizer.fit_transform(texts)
        frequencies = np.asarray(matrix.sum(axis=0)).ravel()
        result = pd.DataFrame({"terme": vectorizer.get_feature_names_out(),
                               "fréquence": frequencies})
        return result.sort_values("fréquence", ascending=False).head(n).reset_index(drop=True)

    def top_ngrams_by_class(self, df: pd.DataFrame, text_col: str = "content",
                            n: int = 20, ngram_range: tuple = (1, 1)) -> Dict[str, pd.DataFrame]:
        """Side-by-side horizontal bar charts of the most frequent n-grams in each class."""
        kind = {1: "termes", 2: "bigrammes", 3: "trigrammes"}.get(ngram_range[0], "n-grammes")
        tops = {name: self.top_ngrams(texts, n, ngram_range)
                for name, texts in self._class_texts(df, text_col).items()}

        fig, axes = plt.subplots(1, len(tops), figsize=(14, 0.3 * n + 2))
        for ax, (name, table), color in zip(axes, tops.items(), ["tab:blue", "tab:red"]):
            data = table.iloc[::-1]
            ax.barh(data["terme"], data["fréquence"], color=color)
            ax.set_title(f"{n} {kind} les plus fréquents — {name}")
            ax.set_xlabel("Fréquence")
        plt.tight_layout()
        plt.show()
        return tops

    def wordclouds_by_class(self, df: pd.DataFrame, text_col: str = "content",
                            max_words: int = 150) -> None:
        """One word cloud per class (requires the wordcloud package)."""
        from wordcloud import WordCloud

        texts = self._class_texts(df, text_col)
        fig, axes = plt.subplots(len(texts), 1, figsize=(12, 6 * len(texts)))
        for ax, (name, series) in zip(axes, texts.items()):
            cloud = WordCloud(width=1200, height=600, background_color="white",
                              max_words=max_words, collocations=False,
                              random_state=42).generate(" ".join(series))
            ax.imshow(cloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Nuage de mots — {name}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def tfidf_discriminative_terms(self, X_tfidf, y: pd.Series, feature_names: np.ndarray,
                                   n: int = 20) -> pd.DataFrame:
        """
        Ranks terms by the difference of their mean TF-IDF weight between classes.
        Positive difference -> more typical of FAKE (1); negative -> more typical of REAL (0).
        """
        y = np.asarray(y)
        fake_mean = np.asarray(X_tfidf[y == 1].mean(axis=0)).ravel()
        real_mean = np.asarray(X_tfidf[y == 0].mean(axis=0)).ravel()
        diff = fake_mean - real_mean

        order = np.argsort(diff)
        table = pd.concat([
            pd.DataFrame({"terme": feature_names[order[::-1][:n]],
                          "écart_tfidf": diff[order[::-1][:n]], "classe": self.label_names[1]}),
            pd.DataFrame({"terme": feature_names[order[:n]],
                          "écart_tfidf": diff[order[:n]], "classe": self.label_names[0]}),
        ], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 0.3 * n + 2))
        for ax, (name, color) in zip(axes, [(self.label_names[0], "tab:blue"),
                                            (self.label_names[1], "tab:red")]):
            data = table[table["classe"] == name].copy()
            data["poids"] = data["écart_tfidf"].abs()
            data = data.sort_values("poids")
            ax.barh(data["terme"], data["poids"], color=color)
            ax.set_title(f"Termes les plus caractéristiques — {name}")
            ax.set_xlabel("Écart de TF-IDF moyen entre les classes")
        plt.tight_layout()
        plt.show()
        return table

    def plot_linear_coefficients(self, coef: np.ndarray, feature_names: np.ndarray,
                                 n: int = 20, model_name: str = "SVM") -> pd.DataFrame:
        """Most influential terms of a linear model (positive -> FAKE, negative -> REAL)."""
        order = np.argsort(coef)
        idx = np.concatenate([order[:n], order[-n:]])
        table = pd.DataFrame({"terme": feature_names[idx], "coefficient": coef[idx]})

        colors = ["tab:blue" if c < 0 else "tab:red" for c in table["coefficient"]]
        plt.figure(figsize=(8, 0.25 * 2 * n + 2))
        plt.barh(table["terme"], table["coefficient"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Termes les plus influents — {model_name} (bleu → REAL, rouge → FAKE)")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        plt.show()
        return table.sort_values("coefficient", ascending=False).reset_index(drop=True)