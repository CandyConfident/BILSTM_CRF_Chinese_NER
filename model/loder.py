
import numpy as np
import codecs

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class LoadDataset(object):

    """Class that iterates over the Datasets,


    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split('\t')
                    if(len(ls)>1):
                        word, tag = ls[0],ls[1]
                    else:
                        word = ls[0]
                        tag = 'O'
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with codecs.open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        k_to_idx = dict()
        idx_to_k = dict()
        with codecs.open(filename,mode='r',encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()

                k_to_idx[word] = idx
                idx_to_k[idx] = word
    except IOError:
        raise MyIOError(filename)
    return k_to_idx, idx_to_k


def save_word2vec(word_to_id, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        word_to_id: dictionary word_to_id[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.random.uniform(-0.04, 0.04, (len(word_to_id), dim))
    # embeddings = np.zeros([len(vocab), dim])
    emb_invalid = 0
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            if len(line) == dim+1:
                word = line[0]
                embedding = [float(x) for x in line[1:]]

                if word in word_to_id:
                    word_idx = word_to_id[word]
                    embeddings[word_idx] = np.asarray(embedding)
            else:
                emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def load_word2vec(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)

