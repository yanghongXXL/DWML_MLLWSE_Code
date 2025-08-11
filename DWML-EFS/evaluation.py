from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import BRkNNbClassifier
from skmultilearn.adapt import MLARAM
from skmultilearn.ensemble import RakelO
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn.metrics as metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier


from mlmetrics import microfscore



def classify(X_train, y_train, X_test, y_test, classifier, mtrs):
    results = {}
    # remove labels with only one value
    labels1 = [np.unique(t) for t in np.transpose(y_train)]
    labels2 = [np.unique(t) for t in np.transpose(y_test)]
    non_bin_idx1 = [i for i in range(len(labels1)) if len(labels1[i]) < 2]
    non_bin_idx2 = [i for i in range(len(labels2)) if len(labels2[i]) < 2]
    non_bin_idx = list(set(non_bin_idx1 + non_bin_idx2))
    y_train = np.delete(y_train, non_bin_idx, axis=1)
    y_test = np.delete(y_test, non_bin_idx, axis=1)

    # 添加日志输出，查看传入的分类器名称
    print("使用的分类器：", classifier)

    if classifier == 'MLKNN':
        clf = BRkNNbClassifier(k=5)
    elif classifier == 'MLARAM':
        clf = MLARAM(threshold=0.05, vigilance=0.95)
    elif classifier == 'RakelOnb':
        clf = RakelO(base_classifier=GaussianNB(), base_classifier_require_dense=[True, True], labelset_size=3, model_count=8)
    elif classifier == 'RakelOrf':
        clf = RakelO(base_classifier=RandomForestClassifier(), base_classifier_require_dense=[True, True], labelset_size=3, model_count=8)
    elif classifier == 'RakelOsvc':
        clf = RakelO(base_classifier=SVC(), base_classifier_require_dense=[True, True], labelset_size=3, model_count=8)
    elif classifier == 'BinaryRelevanceSvc':
        clf = BinaryRelevance(classifier=SVC())
    elif classifier == 'BinaryRelevanceRf':
        clf = BinaryRelevance(classifier=RandomForestClassifier())
    elif classifier == 'ClassifierChainRf':
        clf = ClassifierChain(classifier=RandomForestClassifier())
    elif classifier == 'ClassifierChainSvc':
        clf = ClassifierChain(classifier=SVC())
    elif classifier == 'LabelPowersetRf':
        clf = LabelPowerset(classifier=RandomForestClassifier())
    elif classifier == 'LabelPowersetSvc':
        clf = LabelPowerset(classifier=SVC())
    else:
        # 如果没有匹配到任何已知的分类器，抛出异常或者使用默认分类器
        raise ValueError("未知的分类器类型: {}".format(classifier))
        # 或者设定一个默认分类器
        # clf = SomeDefaultClassifier()

    prediction = clf.fit(X_train, y_train).predict(X_test)


    for m in mtrs:
        if m == 'hamming':
            results[m] = metrics.hamming_loss(y_test, prediction.toarray())

        if m == 'ranking':
            results[m] = metrics.label_ranking_loss(y_test, prediction.toarray())

        if m == 'coverage':
            results[m] = metrics.coverage_error(y_test, prediction.toarray())

        if m == 'averageprecisionscore':
            results[m] = metrics.average_precision_score(y_test, prediction.toarray())

        if m == 'f1_score':
            results[m] = metrics.f1_score(y_test, prediction.toarray(), average='weighted')

        if m == 'accuracyscore':
            results[m] = metrics.accuracy_score(y_test, prediction.toarray())

        if m == 'jaccardscore':
            results[m] = metrics.jaccard_score(y_test, prediction.toarray(), average='weighted')

        if m == 'microfscore':
            results[m] = microfscore(y_test, prediction.toarray())

        if m == 'zeroone':
            results[m] = metrics.zero_one_loss(y_test, prediction.toarray())

    return results


