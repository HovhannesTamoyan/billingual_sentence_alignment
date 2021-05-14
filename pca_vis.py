import argparse
from typing import List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from emb_extractor import EmbExtractor
from emb_extractor import extract_file_name

from dataset import Dataset


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

    dataset = Dataset(args.dataset_name)
    lines = dataset.extract_lines(args.fit_num_sentences, args.langs, args.seed)

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

            # lang for LASER
            embs = emb_extractor.extract_emb(list(batch_lines), lang)
            lang_emb.append(embs)

        lang_emb = np.concatenate(lang_emb, 0)
        if args.center_zero:
            lang_embs.append(lang_emb - lang_emb.mean(axis=0, keepdims=True))
        else:
            lang_embs.append(lang_emb)

    embs = np.concatenate(lang_embs, 0)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(embs)

    labels_df = pd.DataFrame(data=np.array(lang_labels), columns=['lang'])

    ids_draw: List[int] = random.sample(range(args.fit_num_sentences), args.draw_num_sentences)
    ids_draw_all: List[int] = [i+j*args.fit_num_sentences for j in range(len(args.langs)) for i in ids_draw]

    draw_principalComponents = []
    for id in ids_draw_all:
        draw_principalComponents.append(principalComponents[id])

    draw_principalComponents = np.array(draw_principalComponents)

    principalDf = pd.DataFrame(data=draw_principalComponents, columns=['pc 1', 'pc 2'])
    ids_df = pd.DataFrame(data=np.array(ids_draw*len(args.langs)), columns=['id'])

    finalDf = pd.concat([principalDf, labels_df, ids_df], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(args.model_name.upper())

    for lang in args.langs:
        indicesToKeep = finalDf['lang'] == lang
        ax.scatter(finalDf.loc[indicesToKeep, 'pc 1'],
                   finalDf.loc[indicesToKeep, 'pc 2'],
                   s=50)

    ax.legend(args.langs)

    if len(args.langs) == 2:
        for id in finalDf['id']:
            record = finalDf.loc[finalDf['id'] == id]
            plt.plot(list(record['pc 1']), list(record['pc 2']), color='k', alpha=0.1)
    else:
        for id in finalDf['id']:
            record = finalDf.loc[finalDf['id'] == id]
            pc1 = list(record['pc 1'])
            pc2 = list(record['pc 2'])
            pc1.append(pc1[0])
            pc2.append(pc2[0])
            plt.plot(pc1, pc2, color='k', alpha=0.1)

    ax.grid()

    file_name = extract_file_name(output_dir=args.output_dir, model_name=args.model_name, 
                                  fp16=args.fp16, pooling=args.pooling, without_encoding=args.without_encoding, 
                                  use_mlm_head=args.use_mlm_head, use_mlm_head_without_layernorm=args.use_mlm_head_without_layernorm, center_zero=args.center_zero, center_zero_by_lang=None)

    file_name = file_name + "." + "-".join(args.langs)
    plt.savefig(f"{file_name}.{args.fit_num_sentences}.{args.draw_num_sentences}.{args.seed}.png")
