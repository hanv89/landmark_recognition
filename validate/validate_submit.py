
import argparse
import csv
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--truth', default='truth.csv', type = str, help = 'truth labels')
parser.add_argument('--submission', default='submission.csv', type = str, help = 'submission labels')
args = parser.parse_args()

print(args.truth)
print(args.submission)

start = time.time()

acc = 0
top3 = 0
total = 0

predictions = dict()
with open(args.submission) as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            predictions[row[0]] = row[1].split()
        line_count+=1
    print(f'Processed {line_count} lines.')

with open(args.truth) as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            if row[1] in predictions.get(row[0],['-1','-1','-1']):
                top3+=1
            if row[1] == predictions.get(row[0],['-1','-1','-1'])[0]:
                acc+=1
            
            total+=1
        line_count+=1
    print(f'Processed {line_count} lines.')

exec_time = time.time() - start
print('Finished after ',exec_time,': acc=', acc, ', top3=', top3, ' / total=', total)