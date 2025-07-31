import torch

from transformers import AutoTokenizer

from word2vec import Word2Vec
from load_corpus import load_corpus
from config import *


if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load corpus
    corpus = load_corpus()

    # declare word2vec
    word2vec = Word2Vec(vocab_size, d_model, window_size, method).to(device)

    # train word2vec
    word2vec.fit(corpus, tokenizer, lr_word2vec, num_epochs_word2vec)

    # save word2vec checkpoint
    torch.save(word2vec.cpu().state_dict(), "word2vec.pt")
