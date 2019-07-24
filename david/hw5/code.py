import gzip
import os
import shutil


def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    with gzip.open(file_name, 'rb') as f_in:
        with open(f_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


import numpy as np
from scipy import sparse
import math

from sklearn.feature_extraction.text import TfidfTransformer

# load
file = 'docword.nips.txt'
dic_file = 'vocab.nips.txt'
if not os.path.exists(file):
    un_gz(file + '.gz')
dic = {}  # 0-base
with open(dic_file, 'r') as f:
    i = 0
    for line in f.readlines():
        word = line.strip('\n')
        dic[i] = word
        i = i + 1

docs = []
D = 0
W = 0
with open(file, 'r') as f:
    D = int(f.readline().strip('\n'))  # DocSize = 1500 # 0-base
    W = int(f.readline().strip('\n'))  # VocSize = 12419
    if W != len(dic):
        raise ArithmeticError

    NNZ = int(f.readline().strip('\n'))  # All records Size = 746136
    for line in f.readlines():
        rec = [int(x) for x in line.strip('\n').split(' ')]  #
        # rec = (row, col, data) = (NO.DOC NO.WORD TIMES)
        docs.append(rec)

# sparse it is no use ??
docs = sparse.coo_matrix(
    ([rec[2] for rec in docs],
        ([rec[0] - 1 for rec in docs], [rec[1] - 1 for rec in docs
                                    ])),  # row-doc, col-word
    shape=(D, W)).tocsr()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(docs)
# print(tfidf)
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=20, n_iter=7)
# svd.fit_transform(tfidf)

# from sklearn.utils.extmath import randomized_svd
# U, Sigma, VT = randomized_svd(tfidf, 
#                               n_components=20,
#                               n_iter=7,
#                               random_state=None)

from scipy.sparse.linalg import svds

num_components = 20
u, s, vt = svds(tfidf, k=num_components)

s = np.diag(s)

no = 0
for k,v in dic.items():
    if v == 'circuit':
        no = k
        break

vi = np.array([v[no] for v in vt])
scores = u.dot(s).dot(vi)

# np.sort(scores)
order = np.argsort(-scores)
# print(scores[order])

for i in range(5):
    row = np.ravel(tfidf.getrow(order[i]).toarray())
    ind = np.ravel(np.argsort(-row))
    print(row[ind])

    print('Doc '+str(order[i]), end=': ')
    for j in range(5):
        print(dic[ind[j]], row[ind[j]], end=' ')
    print()

