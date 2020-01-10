"""
custom word embeddings in TorchText

@author: Zhenwei Wang
@date: 12/30/2019
"""
import os
import torchtext.vocab as vocab
import torch


def load_custom_embeddings(custom_word_embedding='sgns.weibo.bigram-char'):
    weibo_word_vector = os.path.join('/', 'home', 'wzw', 'pretrained_word_embeddings', custom_word_embedding)
    cache = os.path.join('/', 'home', 'wzw', 'pretrained_word_embeddings', 'cache.' + custom_word_embedding)
    custom_embeddings = vocab.Vectors(name=weibo_word_vector,
                                      cache=cache,
                                      unk_init=torch.Tensor.normal_)
    return custom_embeddings
