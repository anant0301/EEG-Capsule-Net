import pandas as pd


traindf = pd.read_csv('/home/patelanant/CapsuleNet/data/capsulenet/time/train-info.csv')
testdf = pd.read_csv('/home/patelanant/CapsuleNet/data/capsulenet/time/test-info.csv')


for i in range(len(traindf.index)):
    traindf['outfile_name'].iloc[i] = 'train/' + traindf['outfile_name'].iloc[i]

for i in range(len(testdf.index)):
    testdf['outfile_name'].iloc[i] = 'test/' + testdf['outfile_name'].iloc[i]


df = pd.concat([traindf, testdf], axis=0)
df = df.sample(frac=1)

split = int(df.shape[0] * 0.7)
rtraindf = df.iloc[:split]
rtestdf = df.iloc[split:]


rtraindf.to_csv('/home/patelanant/CapsuleNet/data/capsulenet/time/r-train-info.csv', index= False)
rtestdf.to_csv('/home/patelanant/CapsuleNet/data/capsulenet/time/r-test-info.csv', index= False)
print(rtestdf.shape)
print(rtraindf.shape)