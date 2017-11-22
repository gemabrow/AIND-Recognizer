import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    test_set_Xlengths = test_set.get_all_Xlengths()

    for test_X, test_lengths in test_set_Xlengths.values():
        max_prob = float('-inf')
        # dict with k,v: word, word likelihooood
        prob_words = dict()
        best_word = None

        for word, model in models.items():
            try:
                prob_words[word] = model.score(test_X, test_lengths)
            except Exception as e:
                # word not possible in model
                prob_words[word] = float('-inf')
            if prob_words[word] > max_prob:
                # highest likelihood so far for this model
                best_word, max_prob = word, prob_words[word]
        guesses.append(best_word)
        probabilities.append(prob_words)

    return probabilities, guesses
