# this is to check metrics by predictions.json file

import json,re,string,argparse
import collections
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0,0,0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1,precision, recall

def Jaccard_index(context,gold_answers,prediction):
    text=" ".join(word_tokenize(context)).lower()
    gold_answers=" ".join(word_tokenize(gold_answers[0])).lower()
    prediction = " ".join(word_tokenize(prediction)).lower()
    if prediction=='':
        pred_set=set()
    else:
        pred_start = text.find(prediction)
        pred_end = len(text) - (text[::-1].find(prediction[::-1]))
        pred_set = set(list(range(pred_start, pred_end)))
        if pred_start==-1 or pred_end==-1:
            pred_set=set()

    if gold_answers=='':
        gold_start = 0
        gold_end = 0
        gold_set=set()
    else:
        gold_start = text.find(gold_answers)
        gold_end = len(text) - (text[::-1].find(gold_answers[::-1]))
        # gold_start = example.answers[0]['answer_start']
        # gold_end = example.answers[0]['answer_end']
        gold_set = set(list(range(gold_start, gold_end)))
        if gold_start==-1 or gold_end==-1:
            gold_set=set()

    intersection=gold_set.intersection(pred_set)
    union=gold_set.union(pred_set)

    intersection_list=list(intersection)
    union_list=list(union)

    intersection_list.sort()
    union_list.sort()

    if not intersection_list:
        intersection_word=''
    else:
        intersection_word=text[intersection_list[0]:intersection_list[-1] + 1]
    if not union_list:
        union_words=''
    else:
        union_words=text[union_list[0]:union_list[-1]+1]

    intersection_word_length=len(word_tokenize(intersection_word))
    union_word_length=len(word_tokenize(union_words))

    if intersection_word_length==0 and union_word_length==0:
        JI=1
    else:
        JI=intersection_word_length/union_word_length

    return JI


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Data input directory. Converted version by convert_data_format.",
    )

    parser.add_argument(
        "--prediction_dir",
        default=None,
        type=str,
        required=True,
        help="prediction.json file.",
    )
    args = parser.parse_args()

    data_dir=args.data_dir
    with open(data_dir, "r", encoding="utf-8") as reader:
        my_data = json.load(reader)

    pred_dir=args.prediction_dir
    with open(pred_dir, "r", encoding="utf-8") as reader:
        pred_data = json.load(reader)

    id_answer=dict()
    for data in my_data['data']:
        id_answer[data['paragraphs'][0]['qas'][0]['id']]=data

    f1=dict()
    precisions=dict()
    recalls=dict()
    em=dict()
    JI=dict()
    L=dict()

    if len(list(pred_data.keys()))==len(list(id_answer.keys())):
        print('ok')
    else:
        print('some of data is missed')

    pred_len=[]
    gt_len=[]
    for id in tqdm(list(id_answer.keys())):
        gt=id_answer[id]['paragraphs'][0]['qas'][0]['answers'][0]['text']
        pred=pred_data[id].strip()
        pred_len.append(len(word_tokenize(pred)))
        gt_len.append(len(word_tokenize(gt)))
        L[id]=len(word_tokenize(gt))
        f1[id],precisions[id],recalls[id]=compute_f1(gt, pred)
        em[id]=compute_exact(gt, pred)
        JI[id]=Jaccard_index(id_answer[id]['paragraphs'][0]['context'],[gt], pred)
        if JI[id]==0:
            JI[id]=Jaccard_index(id_answer[id]['paragraphs'][0]['context'], [gt], pred[0:-5])



    f1_np = np.array(list(f1.values()))
    recall_np = np.array(list(recalls.values()))
    precision_np = np.array(list(precisions.values()))
    em_np = np.array(list(em.values()))
    JI_np = np.array(list(JI.values()))
    L_np = np.array(list(L.values()))


    print('mean F1=', np.mean(f1_np))
    print('mean Recall=', np.mean(recall_np))
    print('mean Precision=', np.mean(precision_np))
    print('mean EM=', np.mean(em_np))
    print('mean Area Intersection over Union or Jaccard Index=', np.mean(JI_np))

    return



if __name__ == "__main__":
    main()