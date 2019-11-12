from model.config import Config

from model.loder import LoadDataset, write_vocab, load_vocab,save_word2vec,load_word2vec

from model.data_utils import get_vocabs, UNK, PAD, \
    get_char_vocab, get_processing_word,get_embeddings_vocab

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config()
    processing_word = get_processing_word(lowercase=False)

    # Generators
    dev   = LoadDataset(config.filename_dev, processing_word)
    test  = LoadDataset(config.filename_test, processing_word)
    train = LoadDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_embeddings_vocab(config.filename_glove,dim=300)

    vocab = vocab_words | vocab_glove
    vocab = list(vocab)

    vocab.insert(0,PAD)
    vocab.insert(1,UNK)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    word_to_id, _ = load_vocab(config.filename_words)

    save_word2vec(word_to_id, config.filename_glove,
                                config.filename_trimmed, config.dim_word)


    # Build and save char vocab
    # train = CoNLLDataset(config.filename_train)
    # vocab_chars = get_char_vocab(train)
    # write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
