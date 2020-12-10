# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import metrics

def get_all_category_scocre(gt_labels_list,preds_list):
    accuracy = metrics.accuracy_score(gt_labels_list,preds_list)

    precision_score = metrics.precision_score(gt_labels_list,preds_list, average='macro')

    recall_score = metrics.recall_score(gt_labels_list,preds_list, average='macro')
    f1_score = metrics.f1_score(gt_labels_list,preds_list, average='macro')
    return accuracy,precision_score,recall_score,f1_score