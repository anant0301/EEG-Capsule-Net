import pandas as pd
import sys

subid = int(sys.argv[1])

traindf = pd.read_csv('./data/capsulenet/time/r-train-info.csv')
testdf = pd.read_csv('./data/capsulenet/time/r-test-info.csv')

def get_subject_data(indf, subid: int):
    l = []
    for i in indf.index:
        l.append(indf.loc[i]['input_file'].startswith("%d_" % (subid)))
    return l

traindf = traindf[get_subject_data(traindf, subid)]
testdf = testdf[get_subject_data(testdf, subid)]

traindf.to_csv('./data/capsulenet/time/train-subject-%d.csv'%(subid), index= False)
testdf.to_csv('./data/capsulenet/time/test-subject-%d.csv'%(subid), index= False)
