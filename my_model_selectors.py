import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.n_constant
        min_bic = float('inf')

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # try fitting hmm_model with n_states
                model = self.base_model(n)
                # score likelihood
                logL = model.score(self.X, self.lengths)
                # d: number of features/data points
                d = len(self.X[0])
                # since initial distribution estimated,
                # not all parameters are considered free
                p = n ** 2 + 2 * d * n - 1
                # BIC = -2 * log L + p * log N
                # L: the likelihood of the fitted model
                # p: the number of fre parameters
                # N: the number of data points
                # pylint:disable=maybe-no-member
                bic_score = -2 * logL + p * np.log(d)
                if bic_score < min_bic:
                    best_num_components, min_bic = n, bic_score
            except Exception as e:
                pass

        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.n_constant
        max_dic = float('-inf')

        other_words = {word for word in self.words
                       if word != self.this_word}

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                # score likelihood of this_word
                score = model.score(self.X, self.lengths)
                # score X and lengths of all other words and return sum
                sum_other = 0

                for word in other_words:
                    other_x, other_ls = self.hwords[word]
                    sum_other += model.score(other_x, other_ls)
                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                dic_score = score - (sum_other / len(other_words))

                if dic_score > max_dic:
                    best_num_components, max_dic = n, dic_score
            except Exception as e:
                pass

        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = self.n_constant
        max_avg_score = float('-inf')

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                scores = []
                seq = self.sequences
                model = self.base_model(n)

                if len(seq) > 1:
                    folds = min(len(seq), 3)
                    split_method = KFold(n_splits=folds)

                    for train_idx, test_idx in split_method.split(seq):
                        self.X, self.lengths = combine_sequences(train_idx,
                                                                 seq)
                        test_X, test_lengths = combine_sequences(test_idx,
                                                                 seq)
                        # append scores for averaging
                        scores.append(model.score(test_X, test_lengths))
                else:
                    # splitting not possible, just append model score
                    scores.append(model.score(self.X, self.lengths))

                avg_score = np.mean(scores)
                if avg_score > max_avg_score:
                    best_num_components, max_avg_score = n, avg_score
            except Exception as e:
                pass

        return self.base_model(best_num_components)
