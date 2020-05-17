""""
converts rt-polaritydata to format similar to IMDB
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='converts rt-polaritydata to format similar to IMDB')

parser.add_argument('--root','-r', type=str,default='/ksozykinraid/data/nlp/',
                    help='path to folder with datasets')

args = parser.parse_args()

root = args.root 

imdb = pd.read_csv('{}/IMDB/imdb.csv'.format(root))

sentiments = pd.unique(imdb.sentiment)

positive = pd.read_csv("{}/rt-polaritydata/rt-polarity.pos".format(root), sep='delimiter', header=None)
positive['sentiment'] = sentiments[0]
negative = pd.read_csv("{}/rt-polaritydata/rt-polarity.neg".format(root), sep='delimiter', header=None)
negative['sentiment'] = sentiments[1]


rt = pd.concat([positive,negative])
rt.columns = imdb.columns
rt = rt[rt.review.map(len) > 100]
rt.to_csv('{}/rt-polaritydata/rt-polarity.csv'.format(root),index=None)