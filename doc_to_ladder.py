import argparse
from glob import glob
from typing import List, Dict
from bs4 import BeautifulSoup
import os


def remove_latin_char(string):
    while string.find('\xa0') != -1:
        loc = string.find('\xa0')
        if loc != -1:
            if string[loc+1:loc+2] == ' ' or string[loc-1:loc] == ' ':
                string = string.replace('\xa0', '')
            else:
                string = string.replace('\xa0', ' ')
    return string


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--input-files-dir', type=str, default=None)
    parser.add_argument('--src-sgm-file-path', type=str, default=None)
    parser.add_argument('--tgt-sgm-file-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--trim-input-docs', action='store_true')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--tgt', type=str, default=None)

    args = parser.parse_args()

    with open(args.src_sgm_file_path, "r") as f:
        src_sgm_file = f.readlines()

    with open(args.tgt_sgm_file_path, "r") as f:
        tgt_sgm_file = f.readlines()

    docs_names = list()

    src_sgm_docs: Dict[str:List] = {}
    last_doc_name = ''
    for src_sgm_line in src_sgm_file:
        soup = BeautifulSoup(src_sgm_line, 'html.parser')
        if src_sgm_line.startswith('<doc'):
            doc_name = soup.findAll("doc")[0]['docid']
            src_sgm_docs[doc_name] = list()
            last_doc_name = doc_name
            docs_names.append(last_doc_name)
        elif src_sgm_line.startswith('<seg'):
            line = soup.findAll("seg")[0].text
            src_sgm_docs[last_doc_name].append(line)
        else:
            continue

    docs_names.sort()

    tgt_sgm_docs: Dict[str:List] = {}
    last_doc_name = ''
    for tgt_sgm_line in tgt_sgm_file:
        soup = BeautifulSoup(tgt_sgm_line, 'html.parser')
        if tgt_sgm_line.startswith('<doc'):
            doc_name = soup.findAll("doc")[0]['docid']
            tgt_sgm_docs[doc_name] = list()
            last_doc_name = doc_name
        elif tgt_sgm_line.startswith('<seg'):
            line = soup.findAll("seg")[0].text
            tgt_sgm_docs[last_doc_name].append(line)
        else:
            continue

    src_input_docs = glob(os.path.join(args.input_files_dir, f"*.{args.src}-{args.tgt}.{args.src}"))
    tgt_input_docs = glob(os.path.join(args.input_files_dir, f"*.{args.src}-{args.tgt}.{args.tgt}"))
    src_input_docs.sort()
    tgt_input_docs.sort()

    src_docs: Dict[str:List] = {}
    for src_input_doc in src_input_docs:
        src_input_doc_list = list()
        with open(src_input_doc, "r") as f:
            for x in f.readlines():
                x = x.strip()
                x = remove_latin_char(x)
                src_input_doc_list.append(x)

            ids_to_pop = list()
            for src_line_i in range(len(src_input_doc_list)):
                if src_input_doc_list[src_line_i] == '<P>' and src_input_doc_list[src_line_i+1] == '<P>':
                    ids_to_pop.append(src_line_i)
            for i in ids_to_pop:
                src_input_doc_list.pop(i)

            # for src_input_doc_item in src_input_doc_list:
            #     if src_input_doc_item == '<HEADLINE>' or src_input_doc_item == '<P>':
            #         src_input_doc_list.remove(src_input_doc_item)

        doc_name = src_input_doc.split('/')[-1].split('.')[0]
        src_docs[doc_name] = src_input_doc_list

    tgt_docs: Dict[str:List] = {}
    for tgt_input_doc in tgt_input_docs:
        tgt_input_doc_list = list()
        with open(tgt_input_doc, "r") as f:
            for x in f.readlines():
                x = x.strip()
                x = remove_latin_char(x)
                tgt_input_doc_list.append(x)

            ids_to_pop = list()
            for tgt_line_i in range(len(tgt_input_doc_list)):
                if tgt_input_doc_list[tgt_line_i] == '<P>' and tgt_input_doc_list[tgt_line_i+1] == '<P>':
                    ids_to_pop.append(tgt_line_i)
            for i in ids_to_pop:
                tgt_input_doc_list.pop(i)
            # for tgt_input_doc_item in tgt_input_doc_list:
            #     if tgt_input_doc_item == '<HEADLINE>' or tgt_input_doc_item == '<P>':
            #         tgt_input_doc_list.remove(tgt_input_doc_item)

        doc_name = tgt_input_doc.split('/')[-1].split('.')[0]
        tgt_docs[doc_name] = tgt_input_doc_list

    src_ladder: Dict[str:List] = {}
    for doc_name in docs_names:
        src_ladder[doc_name] = list()
        src_idx = 0
        src_tmp_ladder = list()
        src_last_match_idx = -1
        for src_line in src_docs[doc_name]:
            src_match_idx = -1
            for src_doc_item in src_sgm_docs[doc_name]:
                if src_line in src_doc_item:
                    src_match_idx = src_doc_item.index(src_line)

            if len(src_tmp_ladder) > 1 and src_match_idx < src_last_match_idx:
                src_tmp_ladder_str = ", ".join(src_tmp_ladder)
                src_ladder[doc_name].append(f"[{src_tmp_ladder_str}]")
                src_tmp_ladder = list()
                src_last_match_idx = -1

            if src_line in src_sgm_docs[doc_name] or (src_line == '<HEADLINE>' or src_line == '<P>'):
                src_ladder[doc_name].append(f"[{src_idx}]")
                src_idx += 1
            elif src_match_idx != -1:
                if src_match_idx >= src_last_match_idx:
                    src_tmp_ladder.append(str(src_idx))
                    src_idx += 1
                    src_last_match_idx = src_match_idx
            else:
                continue

    tgt_ladder: Dict[str:List] = {}
    for doc_name in docs_names:
        tgt_ladder[doc_name] = list()
        idx = 0
        tgt_tmp_ladder = list()
        last_match_idx = -1
        for tgt_line in tgt_docs[doc_name]:
            match_idx = -1
            for tgt_doc_item in tgt_sgm_docs[doc_name]:
                if tgt_line in tgt_doc_item:
                    match_idx = tgt_doc_item.index(tgt_line)

            if len(tgt_tmp_ladder) > 1 and match_idx < last_match_idx:
                tgt_tmp_ladder_str = ", ".join(tgt_tmp_ladder)
                tgt_ladder[doc_name].append(f"[{tgt_tmp_ladder_str}]")
                tgt_tmp_ladder = list()
                last_match_idx = -1

            if tgt_line in tgt_sgm_docs[doc_name] or (tgt_line == '<HEADLINE>' or tgt_line == '<P>'):
                tgt_ladder[doc_name].append(f"[{idx}]")
                idx += 1
            elif match_idx != -1:
                if match_idx >= last_match_idx:
                    tgt_tmp_ladder.append(str(idx))
                    idx += 1
                    last_match_idx = match_idx
            else:
                continue

    for doc_name in docs_names:
        assert len(src_ladder[doc_name]) == len(tgt_ladder[doc_name])
        with open(f"{args.output_dir}/{doc_name}.{args.src}-{args.tgt}.{args.src}{args.tgt}", "w") as f:
            for src, tgt in zip(src_ladder[doc_name], tgt_ladder[doc_name]):
                f.write(f"{src}:{tgt}\n")

    if args.trim_input_docs:
        trimed_input_files_dir = f"{args.input_files_dir}_trimed"
        if not os.path.exists(trimed_input_files_dir):
            os.makedirs(trimed_input_files_dir)

        for doc_name in docs_names:
            for id in range(len(src_docs[doc_name])-1):
                if src_docs[doc_name][id] == src_sgm_docs[doc_name][-1]:
                    while id+1 < len(src_docs[doc_name]):
                        src_docs[doc_name].pop(id+1)
                    break

            for id in range(len(tgt_docs[doc_name])-1):
                if tgt_docs[doc_name][id] == tgt_sgm_docs[doc_name][-1]:
                    while id+1 < len(tgt_docs[doc_name]):
                        tgt_docs[doc_name].pop(id+1)
                    break

            with open(f"{trimed_input_files_dir}/{doc_name}.{args.src}-{args.tgt}.{args.src}", "w") as f:
                for src_line in src_docs[doc_name]:
                    f.write(src_line)
                    f.write('\n')

            with open(f"{trimed_input_files_dir}/{doc_name}.{args.src}-{args.tgt}.{args.tgt}", "w") as f:
                for tgt_line in tgt_docs[doc_name]:
                    f.write(tgt_line)
                    f.write('\n')


if __name__ == '__main__':
    main()
