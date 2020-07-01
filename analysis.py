import pandas as pd
from nltk.tokenize import word_tokenize

from baselines import TOXIC_CATEGORIES

df = pd.read_csv("data/train.csv")

# Total
print("Length of train set: %d\n" % df.shape[0])

# Distribution
count_labels = {k: int(df[k].sum()) for k in TOXIC_CATEGORIES}
count_labels = {k: v for k, v in sorted(count_labels.items(), key=lambda x: x[1], reverse=True)}

for k, v in count_labels.items():
    print("- %s : %d comments" % (k, v))

print()

# Character length
char_len = df.comment_text.apply(len)

print("Character length:")
print("Average number of characters: %d" % char_len.mean())
print("Median number of characters: %d" % char_len.median())
print("Std number of characters: %d" % char_len.std())
print("Minimum number of characters: %d" % char_len.min())
print("Maximum number of characters: %d" % char_len.max())
print()

# Token counts
token_counts = df.comment_text.apply(lambda x: len(word_tokenize(x)))  # slow processing.

print("Token count:")
print("Average number of tokens: %d" % token_counts.mean())
print("Median number of tokens: %d" % token_counts.median())
print("Std number of tokens: %d" % token_counts.std())
print("Minimum number of tokens: %d" % token_counts.min())
print("Maximum number of tokens: %d" % token_counts.max())
print()
