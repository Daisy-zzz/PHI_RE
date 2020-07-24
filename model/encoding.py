import numpy as np
import preprocess


def embedding():
    data = preprocess.to_encode()
    embedded_data = []
    for line in data:
        embedded_data.append(np.random.randn(len(line), 256))
        print(len(line))
    return data, embedded_data


def pos_embedding():
    pass


def sen_embeddding():
    pass


def word_embedding():
    pass


