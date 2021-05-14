import argparse
from typing import List, Union
import torch
from torch.nn import Identity
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import logging


class EmbExtractor():

    def __init__(self,
                 model_name: str,
                 sentence_transformer: bool,
                 gpu: bool,
                 fp16: bool,
                 pooling: str,
                 without_encoding: bool,
                 use_mlm_head: bool,
                 use_mlm_head_without_layernorm: bool):

        self._sentence_transformer = sentence_transformer
        self._gpu = gpu
        self._fp16 = fp16
        self._pooling = pooling
        self._without_encoding = without_encoding
        self._use_mlm_head = use_mlm_head
        self._use_mlm_head_without_layernorm = use_mlm_head_without_layernorm

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._sentence_transformer:
            self._model = SentenceTransformer(model_name)
        else:
            if self._pooling == "mask" or self._use_mlm_head:
                self._model = AutoModelForMaskedLM.from_pretrained(model_name)
                self._model.config.output_hidden_states = True
            else:
                self._model = AutoModel.from_pretrained(model_name)

        if self._gpu:
            self._model.cuda()
        if self._fp16:
            self._model.half()

    def extract_emb(self,
                    lines: Union[str, List[str]]):

        if not isinstance(lines, list):
            lines = [lines]

        if self._sentence_transformer:
            # Shape: (batch_size, num_embs)
            sentence_embedding = self._model.encode(lines)

            return sentence_embedding
        else:
            encoded_input = self._tokenizer.batch_encode_plus(lines, truncation=True, padding=True, pad_to_multiple_of=8, return_tensors='pt', return_special_tokens_mask=True)
            if self._gpu:
                encoded_input = {
                    k: v.cuda()
                    for k, v
                    in encoded_input.items()
                }

            # Shape: (batch_size, num_tokens, 1)
            special_tokens_mask = (1 - encoded_input.pop("special_tokens_mask").unsqueeze(axis=-1))

            if self._use_mlm_head:
                self._model.lm_head.decoder = Identity()
                if self._use_mlm_head_without_layernorm:
                    self._model.lm_head.lm_head_norm = Identity()

            with torch.no_grad():
                outputs = self._model(**encoded_input)

            if self._use_mlm_head:
                self._pooling = "mask"

            if self._pooling == "mask":
                assert not self._without_encoding
                # Shape: (batch_size, num_tokens, num_embs)
                output = outputs["hidden_states"][-1]

                if self._use_mlm_head:
                    with torch.no_grad():
                        # Shape: (batch_size, num_tokens, num_embs)
                        output = self._model.lm_head(output)
                # Shape: (batch_size, num_embs) - <mask> is the 2nd token
                sentence_embedding = output[:,1,:]
                # ...
            elif self._pooling == "cls":
                # Shape: (batch_size, num_tokens, num_embs)
                output = outputs["last_hidden_state"]
                # Shape: (batch_size, num_embs)
                sentence_embedding = output[:, 0, :]
            else:

                if self._without_encoding:
                    # Shape: (batch_size, num_embs)
                    output = outputs["last_hidden_state"][0] * special_tokens_mask
                else:
                    # Shape: (batch_size, num_tokens, num_embs)
                    output = outputs["last_hidden_state"] * special_tokens_mask

                if self._pooling == 'avg':
                    # Shape: (batch_size, num_embs)
                    output_masked = torch.sum(output, dim=1)
                    # Shape: (batch_size, 1)
                    non_zeros_n = torch.sum(special_tokens_mask, dim=1)

                    # Shape: (batch_size, num_embs)
                    sentence_embedding = output_masked / non_zeros_n
                elif self._pooling == 'max':
                    # Shape: (batch_size, num_embs)
                    output_masked = (output).max(dim=1)

                    # Shape: (batch_size, num_embs)
                    sentence_embedding = output_masked.values
                else:
                    logging.critical(" - pooling method doesnt exists")
                    exit()

            return sentence_embedding.float().cpu().numpy()


def emb_writer(encoder: EmbExtractor,
               input_path: str,
               output_path: str,
               batch_size: int,
               bucketing: bool,
               center_zero: bool,
               center_zero_by_lang: bool = False,
               output_other_path: str = None):

    batches = []

    logging.info(" - Input: reading ...")
    with open(input_path, "r") as overlap_src:
        lines = overlap_src.readlines()
        num_lines = len(lines)

        lines = [line.strip() for line in lines]

        if encoder._pooling == "mask":
            lines = ["<mask> {0}".format(line) for line in lines]

        if bucketing:
            permute = np.argsort([len(line) for line in lines])

            lines = [lines[i] for i in permute]

        progress = tqdm(total=num_lines)
        for start in range(0, num_lines, batch_size):
            end = start + batch_size
            batch_lines = lines[start:end]

            # Shape: (batch_size, num_embs)
            batch_embs = encoder.extract_emb(batch_lines)
            batches.append(batch_embs)
            progress.update(len(batch_lines))

        # Shape: (num_lines, num_embs)
        embs = np.concatenate(batches, axis=0)

        if bucketing:
            new_embs = np.zeros_like(embs)
            new_embs[permute] = embs
        else:
            new_embs = embs

    logging.info(new_embs.shape)

    with open(output_path, "w") as f_output:
        logging.info(" - Output: writing ...")
        if center_zero or center_zero_by_lang:
            if center_zero:
                new_embs -= new_embs.mean(axis=0, keepdims=True)
            else:
                np.save(f"{output_path}._mean", new_embs.mean(axis=0, keepdims=True))
                if output_other_path:
                    other_mean = np.load(output_other_path) - new_embs.mean(axis=0, keepdims=True)
                    new_embs -= other_mean
        new_embs.tofile(f_output)


def extract_file_name(output_dir, model_name, fp16, pooling, center_zero, center_zero_by_lang, use_mlm_head, use_mlm_head_without_layernorm, without_encoding):
    fp16 = ".fp16" if fp16 else ""
    center_zero = ".center_zero" if center_zero else ""
    center_zero_by_lang = ".center_zero_by_lang" if center_zero_by_lang else ""
    use_mlm_head = ".use_mlm_head" if use_mlm_head else ""
    use_mlm_head_without_layernorm = ".use_mlm_head_without_layernorm" if use_mlm_head_without_layernorm else ""
    pooling = f".{pooling}" if pooling else ""
    without_encoding = ".without_encoding" if without_encoding else ""
    model_name = model_name.replace("/", "-").lower()

    return output_dir + f"{model_name}{fp16}{center_zero}{center_zero_by_lang}{pooling}{use_mlm_head}{use_mlm_head_without_layernorm}{without_encoding}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vecalign: Transformer based')

    parser.add_argument('--src_lang', type=str, required=True,
                        help='src language')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='tgt language')
    parser.add_argument('--src_input_path', type=str, required=True,
                        help='src input directory to read overlaped corpus')
    parser.add_argument('--src_output_dir', type=str,
                        help='src output dir to store the .emb file')
    parser.add_argument('--tgt_input_path', type=str, required=True,
                        help='tgt input directory to read overlaped corpus')
    parser.add_argument('--tgt_output_dir', type=str,
                        help='tgt output dir to store the .emb file')

    parser.add_argument('--model_name', type=str, required=False, default="bert-base-multilingual-cased",
                        help='model to be used, default is "bert-base-multilingual-cased"')
    parser.add_argument('--sentence_transformer', action="store_true",
                        help='whether the --model is a Sentence Transformer')
    parser.add_argument('--bert_model', action="store_true",
                        help='whether the --model is a Sentence Transformer')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size, default is 1')
    parser.add_argument('--gpu', action="store_true",
                        help='whether to use GPU')
    parser.add_argument('--fp16', action="store_true",
                        help='whether to use fp16')
    parser.add_argument('--bucketing', action="store_true",
                        help='whether to use bucketing')
    parser.add_argument('--pooling', type=str,
                        help='whether to use avg, max, mask or cls pooling')
    parser.add_argument('--without_encoding', action="store_true",
                        help='whether to pass only the very first layer of the model')
    parser.add_argument('--use_mlm_head', action="store_true",
                        help='whether to use LM head')
    parser.add_argument('--use_mlm_head_without_layernorm', action="store_true",
                        help='whether to use the LM head without LayerNorm')
    parser.add_argument('--center_zero', action="store_true",
                        help='whether to subtract the mean of the tensor from it')
    parser.add_argument('--center_zero_by_lang', action="store_true",
                        help='whether to subtract the mean of the tensor from the other language')

    parser.add_argument('--output_custom_names', action="store_true",
                        help='whether to store output files with custom names')
    parser.add_argument('--src_output_path', type=str, required=True,
                        help='src output file path with custom name')
    parser.add_argument('--tgt_output_path', type=str, required=True,
                        help='tgt output file path with custom name')

    args = parser.parse_args()

    if args.output_custom_names:
        assert args.src_output_path is not None
        assert args.tgt_output_path is not None
    if args.fp16 and not args.gpu:
        raise Exception("gpu is mandatory for fp16")
    if args.sentence_transformer and args.pooling:
        raise Exception("pooling is useless when using Sentence Transformer")
    if args.use_mlm_head_without_layernorm and not args.use_mlm_head:
        raise Exception("--use_mlm_head is mandatory for --use_mlm_head_without_layernorm")
    if args.use_mlm_head and args.pooling:
        logging.info("pooling method is useless")
    if args.center_zero and args.center_zero_by_lang:
        raise Exception("wether use --center_zero or --center_zero_by_lang")

    logging.info(' - Model: loading {}'.format(args.model_name))
    encoder = EmbExtractor(args.model_name, 
                           args.sentence_transformer, 
                           args.gpu, 
                           args.fp16, 
                           args.pooling,
                           args.without_encoding,
                           args.use_mlm_head, 
                           args.use_mlm_head_without_layernorm)

    print(args.output_custom_names)

    if not args.output_custom_names:
        src_output = extract_file_name(args.src_output_dir,
                                       args.model_name, args.fp16,
                                       args.pooling, args.center_zero, args.center_zero_by_lang,
                                       args.use_mlm_head, args.use_mlm_head_without_layernorm, args.without_encoding)

        tgt_output = extract_file_name(args.tgt_output_dir,
                                       args.model_name, args.fp16,
                                       args.pooling, args.center_zero, args.center_zero_by_lang,
                                       args.use_mlm_head, args.use_mlm_head_without_layernorm, args.without_encoding)

        src_output = f"{src_output}.{args.src_lang}"
        tgt_output = f"{tgt_output}.{args.tgt_lang}"
    else:
        src_output = args.src_output_path
        tgt_output = args.tgt_output_path

    emb_writer(encoder, args.src_input_path, f"{src_output}", args.batch_size, args.bucketing, args.center_zero, args.center_zero_by_lang)
    if args.center_zero_by_lang:
        emb_writer(encoder, args.tgt_input_path, f"{tgt_output}", args.batch_size, args.bucketing, args.center_zero, args.center_zero_by_lang, f"{src_output}._mean.npy")
    else:
        emb_writer(encoder, args.tgt_input_path, f"{tgt_output}", args.batch_size, args.bucketing, args.center_zero)
