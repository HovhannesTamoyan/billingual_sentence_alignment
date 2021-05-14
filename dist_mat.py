import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from emb_extractor import EmbExtractor
from emb_extractor import extract_file_name

from dataset import Dataset

from sklearn.metrics.pairwise import pairwise_distances
import numba


@numba.njit
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


@numba.njit
def c_sums(mat: np.ndarray, i: int, j: int, n: int, S: int):
    x_s = np.random.choice(n, size=S)
    y_s = np.random.choice(n, size=S)

    x_terms = np.zeros((S))
    y_terms = np.zeros((S))

    for id, s in enumerate(y_s):
        x_terms[id] = 1 - cosine_similarity_numba(mat[i], mat[s])

    for id, s in enumerate(x_s):
        y_terms[id] = 1 - cosine_similarity_numba(mat[s], mat[j])

    sum1 = np.sum(x_terms)
    sum2 = np.sum(y_terms)

    return sum1, sum2

@numba.njit
def vecalign_pariwise_distances(mat: np.ndarray, S: int):
    n = mat.shape[0]
    n_sent = int(n/2)
    ret = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum1, sum2 = c_sums(mat, i, j, n_sent, S)
            ret[i][j] = ((1-cosine_similarity_numba(mat[i], mat[j])) * n_sent * n_sent)/(sum1 + sum2)

    return ret


def sim_score(mat: np.ndarray,
              metric: str):

    if metric == "vecalign":
        # normalizing - veclaign has n^2 multiplied
        mat = mat/(mat.shape[0]**2)

    sim_score = np.sum(np.diagonal(mat))/len(np.diagonal(mat))
    dis_score = np.sum(np.tril(mat, -1))/np.count_nonzero(np.tril(mat, -1))
    sim_std = np.std(np.diagonal(mat))
    dis_std = np.std(np.tril(mat, -1))

    print(sim_score, dis_score, sim_std, dis_std)
    return sim_score, dis_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA visualisation')

    parser.add_argument('--fit_num_sentences', type=int, required=False, default=1000,
                        help='number of sentences to fit PCA for')
    parser.add_argument('--draw_num_sentences', type=int, required=False, default=12,
                        help='number of sentences to draw PCA for')
    parser.add_argument('--seed', type=int, required=False, default=17,
                        help='seed for picking random sentences from the set')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size, default is 1')
    parser.add_argument('--langs', nargs='+', default=[],
                        help='the list of languages e.g. [en de fr]')
    parser.add_argument('--dataset_name', type=str, required=False, default="xnli",
                        help='name of the dataset to be loaded from "datasets"')
    parser.add_argument('--model_name', type=str, required=False, default="xlm-roberta-large",
                        help='model to be used, default is "xlm-roberta-large"')
    parser.add_argument('--metric', type=str,
                        help='which metric to use for distance measurement')
    parser.add_argument('--sentence_transformer', action="store_true",
                        help='whether the --model is a Sentence Transformer')
    parser.add_argument('--gpu', action="store_true",
                        help='whether to use GPU')
    parser.add_argument('--fp16', action="store_true",
                        help='whether to use fp16')
    parser.add_argument('--pooling', type=str,
                        help='whether to use avg, max or mask pooling')
    parser.add_argument('--without_encoding', action="store_true",
                        help='whether to pass only the very first layer of the model')
    parser.add_argument('--use_mlm_head', action="store_true",
                        help='whether to use LM head')
    parser.add_argument('--use_mlm_head_without_layernorm', action="store_true",
                        help='whether to use the LM head without LayerNorm')
    parser.add_argument('--center_zero', action="store_true",
                        help='whether to subtract the mean of the tensor from it')
    parser.add_argument('--output_dir', type=str,
                        help='dir to store the .png files')

    args = parser.parse_args()

    dataset = Dataset()
    lines = dataset.extract_lines(args.fit_num_sentences)

    emb_extractor = EmbExtractor(args.model_name, args.sentence_transformer,
                                 args.gpu, args.fp16, args.pooling, args.without_encoding, 
                                 args.use_mlm_head, args.use_mlm_head_without_layernorm)

    lang_embs = []
    lang_labels = []
    lang_ids = []

    for lang in args.langs:
        lang_labels += [lang]*args.draw_num_sentences
        ids, sentences = list(zip(*lines[lang]))
        lang_ids += ids
        lang_emb = []

        for start in range(0, args.fit_num_sentences, args.batch_size):
            end = start + args.batch_size
            batch_lines = sentences[start:end]

            embs = emb_extractor.extract_emb(list(batch_lines), lang)
            lang_emb.append(embs)

        lang_emb = np.concatenate(lang_emb, 0)
        if args.center_zero:
            lang_embs.append(lang_emb - lang_emb.mean(axis=0, keepdims=True))
        else:
            lang_embs.append(lang_emb)

    embs = np.concatenate(lang_embs, 0)

    src = lang_embs[0]
    tgt = lang_embs[1]

    src_tgt = np.concatenate((src, tgt), axis=0)

    if args.metric == "vecalign":
        pairw_dist = vecalign_pariwise_distances(src_tgt, S=3)
    else:
        pairw_dist = pairwise_distances(src_tgt, metric=args.metric)

    sim_score, dis_score = sim_score(pairw_dist[int(src_tgt.shape[0]/2):, 0: int(src_tgt.shape[0]/2)], args.metric)

    print(sim_score, dis_score, sim_score-dis_score)

    plt.imshow(pairw_dist, cmap='seismic')
    file_name = extract_file_name(output_dir=args.output_dir, model_name=args.model_name, 
                                  fp16=args.fp16, pooling=args.pooling, without_encoding=args.without_encoding, 
                                  use_mlm_head=args.use_mlm_head, use_mlm_head_without_layernorm=args.use_mlm_head_without_layernorm, center_zero=args.center_zero, center_zero_by_lang=None)
    file_name = file_name + "." + "-".join(args.langs)

    plt.title(args.model_name)
    plt.savefig(f"{file_name}.{args.metric}.png")
