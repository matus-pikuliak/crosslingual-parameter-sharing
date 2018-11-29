def word_normalization(word):
    """
    This is a function that processes the words from datasets into format compatible with embeddings
    files. It should be edited to fit the embeddings we use for various languages.
    """

    if word[0].isnumeric():
        return '#'
    if word.startswith("'") or word.startswith('-'):  # First for English, second for Spanish
        word = word[1:]
    if word == "n't":  # For English
        word = 'not'
    if word.endswith('.') and word != '.':  # Abbreviations
        word = word[:-1]
    return word.lower()