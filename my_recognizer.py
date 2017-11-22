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
    # def train_all_words(features, model_selector):
    #     training = asl.build_training(features)
    #     sequences = training.get_all_sequences()
    #     Xlengths = training.get_all_Xlengths()
    #     model_dict = {}
    #     for word in training.words:
    #         model = model_selector(sequences, Xlengths, word,
    #                         n_constant=3).select()
    #         model_dict[word]=model
    #     return model_dict
    #
    # models = train_all_words(features_ground, SelectorConstant)
    # print("Number of word models returned = {}".format(len(models)))
    # models = dict{'WORD': GaussianHMM model}
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
                best_word, max_prob = word, prob_words[word]
        guesses.append(best_word)
        probabilities.append(prob_words)

    return probabilities, guesses
