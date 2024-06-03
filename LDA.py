import argparse
import time

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from tokenizers.implementations import BertWordPieceTokenizer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-d', "--dataset", type=str, default="sst2")
    args.add_argument('-s', "--text", type=str, default="hello word")
    return args.parse_args()


def preprocessing():
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=lowercase, strip_accents=strip_accents,
        clean_text=clean_text)
    whole_word_list = []
    whole_word_dir = {}
    docs = []
    currentDocument = []
    currentWordId = 0
    if args.dataset == 'ag_news':
        dataset = load_dataset('ag_news')
    if args.dataset == 'sst2':
        dataset = load_dataset('glue', 'sst2')
    if args.dataset == 'cola':
        dataset = load_dataset('glue', 'cola')
    if args.dataset == 'imdb':
        dataset = load_dataset('imdb')
    if args.dataset == 'imdb':
        dataset = load_dataset('imdb')
    if args.dataset == 'de':
        dataset = load_dataset('mteb/mtop_intent', 'de')
    if args.dataset == 'en':
        dataset = load_dataset('mteb/mtop_intent', 'en')
    if args.dataset == 'es':
        dataset = load_dataset('mteb/mtop_intent', 'es')
    if args.dataset == 'fr':
        dataset = load_dataset('mteb/mtop_intent', 'fr')
    if args.dataset == 'hi':
        dataset = load_dataset('mteb/mtop_intent', 'hi')
    if args.dataset == 'th':
        dataset = load_dataset('mteb/mtop_intent', 'th')
    if args.dataset in ['ag_news', 'imdb']:
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        datasets = [train_dataset, test_dataset]
        flag = 'text'
    if args.dataset in ['sst2', 'cola', 'de', 'en', 'es', 'fr', 'hi', 'th']:
        train_dataset = dataset['train']
        validation_dataset = dataset['validation']
        test_dataset = dataset['test']
        datasets = [train_dataset, validation_dataset, test_dataset]
        flag = 'sentence'
    if args.dataset in ['de', 'en', 'es', 'fr', 'hi', 'th']:
        flag = 'text'
    for dataset in datasets:
        for example in dataset:
            id_list = []
            whole_word = ""
            document = example[flag]
            encoding = tokenizer.encode(document, add_special_tokens=False)
            tokens = encoding.tokens
            ids = encoding.ids
            for token, token_ids in zip(tokens, ids):
                if id_list:
                    if token.startswith("##"):
                        whole_word += token[2:]
                        id_list.append(token_ids)
                    else:
                        if len(id_list) < 5:
                            num_zeros = 5 - len(id_list)
                            id_list.extend([0] * num_zeros)
                        id_list_tuple = tuple(id_list)
                        currentDocument.append(id_list_tuple)
                        if whole_word not in whole_word_list:
                            whole_word_list.append(whole_word)
                            whole_word_dir[id_list_tuple] = currentWordId
                            currentWordId += 1
                        id_list = [token_ids]
                        whole_word = token
                else:
                    id_list = [token_ids]
                    whole_word = token
            if id_list:
                num_zeros = 5 - len(id_list)
                id_list.extend([0] * num_zeros)
                id_list_tuple = tuple(id_list)
                currentDocument.append(id_list_tuple)
                if whole_word not in whole_word_list:
                    whole_word_list.append(whole_word)
                    whole_word_dir[id_list_tuple] = currentWordId
                    currentWordId += 1
            docs.append(currentDocument)
            currentDocument = []
    return docs, whole_word_list, whole_word_dir


def randomInitialize():
    for d, doc in enumerate(docs):
        zCurrentDoc = []
        for w in doc:
            w_id = words_idx[w]
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w_id]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            ndz[d, z] += 1
            nzw[z, w_id] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)


def gibbsSampling():

    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            w_id = words_idx[w]
            z = Z[d][index]
            ndz[d, z] -= 1
            nzw[z, w_id] -= 1
            nz[z] -= 1
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w_id]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d][index] = z
            ndz[d, z] += 1
            nzw[z, w_id] += 1
            nz[z] += 1


def perplexity():
    nd = np.sum(ndz, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for w in doc:
            w_id = words_idx[w]
            ll = ll + np.log(((nzw[:, w_id] / nz) * (ndz[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


if __name__ == "__main__":
    args = parse_args()
    data_cfg = OmegaConf.load(f"./cfg/{args.dataset}.yml")
    vocab_file = data_cfg.vocab.tokenizer.vocab
    lowercase = data_cfg.vocab.tokenizer.lowercase
    strip_accents = data_cfg.vocab.tokenizer.strip_accents
    clean_text = data_cfg.vocab.tokenizer.clean_text
    alpha = 0.04
    beta = 0.01
    iterationNum = 10
    Z = []
    K = 64
    docs, words, words_idx = preprocessing()
    N = len(docs)
    M = len(words)
    ndz = np.zeros([N, K]) + alpha
    nzw = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta
    randomInitialize()
    for i in range(0, iterationNum):
        gibbsSampling()
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity())
    nzw_t = np.transpose(nzw)
    sum_nzw = np.sum(nzw_t, axis=1)
    nzw_normalized = np.divide(nzw_t, sum_nzw[:, np.newaxis])
    word_topic_dict = {}
    for word, word_id in words_idx.items():
        topic = nzw_normalized[word_id]
        word_topic_dict[word] = topic
    np.save(f"D:\pycode\LDAdata\{args.dataset}K={K}.npy", word_topic_dict)
