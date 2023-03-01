
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
    assert dataloader(path, mode='n') == ['550', '-1.20335121324857', '1.77464096941025', '0.299299909519384', '-0.418264753573692', '0.484617042285845', '0.071522617576607', '0.0468992068106925', '-1.49254302202968', '0.308864546636991', '0.271294633423322', '-1.53280793963207', '-0.702163217867801', '-0.25811598494555', '-0.485219687472878', '1.05728019068849', '0.787659982979083', '-0.357700747003316', '0.382133003715874', '0.350549744063767', '0.0355676778398812', '1.10005872737751', '-1.56111785243318', '0.0540783239866291', '-1.11910287536837', '0.145057009360543', '0.182842662146484', '0.528715800532048', '0.218047946878206', '2.67', '0']
