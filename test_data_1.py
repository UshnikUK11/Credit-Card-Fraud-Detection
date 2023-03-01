
import pytest
import csv
import random

def dataloader(path, mode='n'):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # print(csv_reader)
        n = random.randint(1, 284000)

        #print(n)
        for i, row in enumerate(csv_reader):
            if i == n and mode=='r':
                data = row
                #print(data)
                #print(data[30])
                break
            elif (i == 624 and mode=='f'):
                data = row
                #print(data)
                break
            elif (i == 732 and mode == 'n'):
                data = row
                #print(data)
                break


        return data

def test_method1():
    path = r"/home/aritra/creditcard.csv"
    assert dataloader(path, mode='f') == ['472', '-3.0435406239976', '-3.15730712090228', '1.08846277997285', '2.2886436183814', '1.35980512966107', '-1.06482252298131', '0.325574266158614', '-0.0677936531906277', '-0.270952836226548', '-0.838586564582682', '-0.414575448285725', '-0.503140859566824', '0.676501544635863', '-1.69202893305906', '2.00063483909015', '0.666779695901966', '0.599717413841732', '1.72532100745514', '0.283344830149495', '2.10233879259444', '0.661695924845707', '0.435477208966341', '1.37596574254306', '-0.293803152734021', '0.279798031841214', '-0.145361714815161', '-0.252773122530705', '0.0357642251788156', '529', '1']
