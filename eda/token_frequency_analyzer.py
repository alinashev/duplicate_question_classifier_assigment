import matplotlib.pyplot as plt
import pandas as pd


class TokenFrequencyAnalyzer:
    """
    Analyzes token frequency from a list of tokens and provides utilities
    to inspect, visualize, and export the frequency distribution.

    Args:
        list_of_tokens (list of str): A list of tokens (e.g., words or subwords).
    """
    def __init__(self, list_of_tokens):
        self.freq_dict = self._count_tokens(list_of_tokens)

    def _count_tokens(self, tokens):
        """
            Counts the frequency of each token in the input list.

            Args:
                tokens (list of str): List of tokens to count.

            Returns:
                dict: Dictionary mapping each token to its frequency.
        """
        freq = {}
        for token in tokens:
            if token:
                freq[token] = freq.get(token, 0) + 1
        return freq

    def top_n(self, n=20):
        """
            Returns the top-N most frequent tokens.

            Args:
                n (int, optional): Number of top tokens to return. Defaults to 20.

            Returns:
                list of tuple: List of (token, count) pairs sorted by frequency descending.
        """
        sorted_items = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def to_dataframe(self):
        """
        Converts the frequency dictionary to a sorted Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns ["token", "count"], sorted by count descending.
        """
        return pd.DataFrame(list(self.freq_dict.items()), columns=["token", "count"]).sort_values(by="count",
                                                                                                  ascending=False)

    def plot_distribution(self, loglog=False, top=None):
        """
        Plots the frequency distribution of tokens.

        Args:
            loglog (bool, optional): Whether to plot using log-log scale. Defaults to False.
            top (int, optional): Show only the top-N ranked tokens. If None, shows all.

        Returns:
            None
        """
        values = sorted(self.freq_dict.values(), reverse=True)
        if top:
            values = values[:top]
        plt.figure(figsize=(10, 5))
        plt.plot(values)
        if loglog:
            plt.yscale("log")
            plt.xscale("log")
            plt.title("Розподіл частоти токенів (логарифмічна шкала)")
        else:
            plt.title("Розподіл частоти токенів")
        plt.xlabel("Ранг")
        plt.ylabel("Частота")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
