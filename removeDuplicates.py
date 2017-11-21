from collections import OrderedDict

inFile = './Data/tweet_data_v2.txt'
outFile= './Data/tweet_data_v2_NoDup.txt'
with open('%s' % inFile) as fin:
    lines = (line.rstrip() for line in fin)
    unique_lines = OrderedDict.fromkeys((line for line in lines if line))

# print(len(lines))
print(len(unique_lines.keys()))
