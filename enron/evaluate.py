from collections import defaultdict
from enum import Enum, auto
import json
import numpy as np
import operator
import os
import pandas as pd
import pickle
import random
import re
from sklearn import metrics
import spacy
from spacy import displacy # v2.0
import sys

class FeatureName(Enum):
    VERB = auto()
    FOLLOWING_POS = auto()
    FOLLOWING_POSTAG = auto()
    CHILD_DEP = auto()
    PARENT_DEP = auto()
    CHILD_POS = auto()
    CHILD_POSTAG = auto()
    PARENT_POS = auto()
    PARENT_POSTAG = auto()

def json_to_tsv(json_f, tsv_f):
    with open(os.path.join(os.path.dirname(__file__), json_f), 'r') as rFile:
        notes = json.load(rFile)

    # only create tsv if it doesn't already exist
    if not os.path.exists(tsv_f):
        for note in notes:
            n_type = note['Type']
            n_content = note['Content']
            # note: had a debate over whether to escape newlines
            # or split newlines as new content (going with latter)

            # https://stackoverflow.com/a/15392758
            # n_content = n_content.encode("unicode_escape").decode("utf-8")
            for split in n_content.splitlines():
                split = split.strip().replace('\t', ' ')
                if not split:
                    continue

                with open(tsv_f, 'a', encoding='utf-8') as wFile:
                    if n_type == 'ToDo':
                        split += '\tpos'
                    else:

                        split += '\tneg'

                    wFile.write(split)
                    wFile.write('\n')

def read_tasks(tsv_f):
    df = pd.read_csv(tsv_f, sep='\t', header=None, names=['Text', 'Label'])
    return [tuple(x) for x in df.to_records(index=False)]

def load_model():
    print("Importing the model from model.pkl")
    with open(os.path.join(os.path.dirname(__file__), '../model.pkl'), 'rb') as f:
        return pickle.load(f)

def init_spacy():
    # slow
    return spacy.load('en')

def featurize(d):
    s_features = defaultdict(int)
    for idx, token in enumerate(d):
        if re.match(r'VB.?', token.tag_) is not None: # note: not using token.pos == VERB because this also includes BES, HVS, MD tags
            s_features[FeatureName.VERB.name] += 1
            # FOLLOWING_POS
            # FOLLOWING_POSTAG
            next_idx = idx + 1
            if next_idx < len(d):
                s_features[f'{FeatureName.FOLLOWING_POS.name}_{d[next_idx].pos_}'] += 1
                s_features[f'{FeatureName.FOLLOWING_POSTAG.name}_{d[next_idx].tag_}'] += 1
            # PARENT_DEP
            # PARENT_POS
            # PARENT_POSTAG
            '''
            "Because the syntactic relations form a tree, every word has exactly one head.
            You can therefore iterate over the arcs in the tree by iterating over the words in the sentence."
            https://spacy.io/docs/usage/dependency-parse#navigating
            '''
            if (token.head is not token):
                s_features[f'{FeatureName.PARENT_DEP.name}_{token.head.dep_.upper()}'] += 1
                s_features[f'{FeatureName.PARENT_POS.name}_{token.head.pos_}'] += 1
                s_features[f'{FeatureName.PARENT_POSTAG.name}_{token.head.tag_}'] += 1
            # CHILD_DEP
            # CHILD_POS
            # CHILD_POSTAG
            for child in token.children:
                s_features[f'{FeatureName.CHILD_DEP.name}_{child.dep_.upper()}'] += 1
                s_features[f'{FeatureName.CHILD_POS.name}_{child.pos_}'] += 1
                s_features[f'{FeatureName.CHILD_POSTAG.name}_{child.tag_}'] += 1
    return dict(s_features)

def predict(classifier, gold, prob=True):
    if (prob is True):
        predictions = classifier.prob_classify_many([fs for (t, fs, ll) in gold])
    else:
        predictions = classifier.classify_many([fs for (t, fs, ll) in gold])
    return list(zip([t for (t, fs, ll) in gold], predictions, [ll for (t, fs, ll) in gold], [fs for (t, fs, ll) in gold]))

def accuracy(predicts, prob=True):
    if (prob is True):
        correct = [label == prediction.max() for (task, prediction, label, fs) in predicts]
    else:
        correct = [label == prediction for (task, prediction, label, fs) in predicts]

    if correct:
        return sum(correct) / len(correct)
    else:
        return 0

def confusion_matrix(predict, prob=True, print_layout=False):
    tasks, predictions, labels, fs = zip(*predict)
    if print_layout is True:
        print('Layout\n[[tn   fp]\n [fn   tp]]\n')
    if prob is True:
        return metrics.confusion_matrix(labels, [p.max() for p in predictions])
    else:
        return metrics.confusion_matrix(labels, predictions)

def classification_report(predict, prob=True):
    tasks, predictions, labels, fs = zip(*predict)
    if prob is True:
        return metrics.classification_report(labels, [p.max() for p in predictions])
    else:
        return metrics.classification_report(labels, predictions)

def preds_to_df(preds):
    preds_df = pd.DataFrame.from_records(preds, columns=['candidate_task', 'prob_dist', 'label', 'features'])
    preds_df['pred'] = [p.max() for p in preds_df['prob_dist']]
    preds_df['pos_dist'] = [p.prob('pos') for p in preds_df['prob_dist']]
    preds_df['neg_dist'] = [p.prob('neg') for p in preds_df['prob_dist']]
    preds_df['max_dist'] = [p.prob(p.max()) for p in preds_df['prob_dist']]

    return preds_df

def print_full_df_column(preds_df, column_names, include_header = False):
    if not isinstance(column_names, list):
        column_names = [column_names]

    return preds_df[column_names].to_csv(sys.stdout, index=False, header=include_header, sep='\t')

def get_labels_enum(preds_df):
    return Enum('Labels', list(preds_df['prob_dist'][0].samples()))

class SamplePredictions(Enum):
    RAND = auto()
    RAND_CORRECT = auto()
    RAND_INCORRECT = auto()
    MOST_INCORRECT = auto()
    MOST_CORRECT = auto()
    MOST_INCORRECT_POS = auto()
    MOST_INCORRECT_NEG = auto()
    MOST_CORRECT_POS = auto()
    MOST_CORRECT_NEG = auto()
    MOST_UNCERTAIN = auto()
    WITH_PROB = auto()

def rand_by_mask(mask, n):
    return np.random.choice(np.where(mask)[0], n, replace=False)

def rand_by_correct(preds_df, n = 0, is_correct = True):
    if n == 0:
        n = len(preds_df)
    return preds_df.iloc[rand_by_mask(([p.max() for p in preds_df['prob_dist']] == preds_df['label'].values)==is_correct, n)]

def most_by_correct(preds_df, label=None, n=0, is_correct=True):
    LABELS = get_labels_enum(preds_df)

    correct_mask = ([p.max() for p in preds_df['prob_dist']] == preds_df['label'].values)==is_correct
    if label is not None:
        target_label_mask = preds_df['label'].values == label
        mask = correct_mask & target_label_mask
        label_options = [name for name, member in LABELS.__members__.items()]
        # note the 2-class assumption
        other_label = label_options[abs(label_options.index(label)-1)]
        # if is_correct, label dist is what matters; if incorrect, look at other_label dist
        dist_column = label + '_dist' if is_correct == True else other_label + '_dist'
    else:
        mask = correct_mask
        dist_column = 'max_dist'

    if n == 0:
        return preds_df.iloc[np.where(mask)].sort_values(dist_column, ascending=False)
    else:
        return preds_df.iloc[np.where(mask)].sort_values(dist_column, ascending=False)[:n]

def most_uncertain(preds_df, n=0):
    LABELS = get_labels_enum(preds_df)

    dist_column = LABELS.pos.name + '_dist' # doesn't matter which

    if n == 0:
        return preds_df.iloc[np.argsort(np.abs(preds_df[dist_column]-0.5))]
    else:
        return preds_df.iloc[np.argsort(np.abs(preds_df[dist_column]-0.5))][:n]

def get_sample_predictions(preds_df, n = 0, sample_type = SamplePredictions.RAND):
    if not isinstance(sample_type, SamplePredictions):
        raise ValueError('sample_type must be an instance of SamplePredictions')

    LABELS = get_labels_enum(preds_df)

    if sample_type == SamplePredictions.RAND:
        mask = len(preds_df)
        if n == 0:
            # return all samples in random order
            n = mask
        return preds_df.iloc[np.random.choice(mask, n)]

    if sample_type == SamplePredictions.RAND_CORRECT:
        return rand_by_correct(preds_df, n)

    if sample_type == SamplePredictions.RAND_INCORRECT:
        return rand_by_correct(preds_df, n, is_correct=False)

    if sample_type == SamplePredictions.MOST_CORRECT:
        return most_by_correct(preds_df, n=n)

    if sample_type == SamplePredictions.MOST_INCORRECT:
        return most_by_correct(preds_df, n=n, is_correct=False)

    if sample_type == SamplePredictions.MOST_CORRECT_POS:
        return most_by_correct(preds_df, LABELS.pos.name, n)

    if sample_type == SamplePredictions.MOST_CORRECT_NEG:
        return most_by_correct(preds_df, LABELS.neg.name, n)

    if sample_type == SamplePredictions.MOST_INCORRECT_POS:
        return most_by_correct(preds_df, LABELS.pos.name, n, is_correct=False)

    if sample_type == SamplePredictions.MOST_INCORRECT_NEG:
        return most_by_correct(preds_df, LABELS.neg.name, n, is_correct=False)

    if sample_type == SamplePredictions.MOST_UNCERTAIN:
        return most_uncertain(preds_df, n)

    raise NotImplemented(sample_type + ' not implemented')

def aggregate_features(preds_df, n=0):
    f_dicts = []
    for i in range(len(most_incorrect)):
        f_dict = pd.DataFrame.from_dict(preds_df[['features']].iloc[i].values[0], orient='index').reset_index()
        f_dict.columns = ['feature', 'count']
        f_dicts.append(f_dict)

    agg = pd.concat(f_dicts).groupby('feature').sum().sort_values('count', ascending=False).reset_index()

    if n <= 0:
        return agg
    else:
        return agg[:n]

class DisplacyStyles(Enum):
    dep = auto()
    ent = auto()

def render_displacy(preds_df, styles):
    NLP = init_spacy()

    if not isinstance(styles, list):
        styles = [styles]

    for i in range(len(preds_df)):
        sent = preds_df[['candidate_task']].iloc[i].values[0]
        doc = NLP(sent)
        for style in styles:
            if not isinstance(style, DisplacyStyles):
                raise ValueError('styles must be a list of DisplacyStyles')
            displacy.render(doc, style=style.name, jupyter=True)
        print([token.tag_ for token in doc])
        print('-----')
        print()

if __name__ == '__main__':
    CLF = load_model()

    JSON_FILE = 'notes.json'
    TSV_FILE = 'notes.tsv'
    TSV_CLEANED_FILE = 'notes.cleaned.tsv'

    print('Loading test data')
    # use cleaned tsv if it exists
    if not os.path.exists(TSV_CLEANED_FILE):
        print(TSV_FILE)
        json_to_tsv(JSON_FILE, TSV_FILE)
        TASKS = read_tasks(TSV_FILE)
    else:
        print(TSV_CLEANED_FILE)
        TASKS = read_tasks(TSV_CLEANED_FILE)

    print('Featurizing test data')
    NLP = init_spacy()
    testing_set = [(task[0], featurize(NLP(task[0])), task[1]) for task in TASKS]

    print('Evaluating')
    preds = predict(CLF, testing_set)

    print()
    print("Accuracy:", accuracy(preds))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(preds, print_layout=True))
    print()
    print("Classification report:")
    print(classification_report(preds))

    preds_df = preds_to_df(preds)

    preds_filtered = np.array(preds)[[len(p[3]) > 0 for p in preds]]
    print()
    print ("Accuracy (on candidates with 1+ feature):", accuracy(preds_filtered))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(preds_filtered))
    print()
    print("Classification report:")
    print(classification_report(preds_filtered))

    #print(get_sample_predictions(preds_df, sample_type=SamplePredictions.MOST_INCORRECT, n=20))
