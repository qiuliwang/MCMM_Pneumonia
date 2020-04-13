'''
provided by LUNA16
evaluation script
'''

import csv
import platform

def writeTXT(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line+'\n')

def writeCSV(filename, lines):
    with open(filename, "wb") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)
        # with open(filename, "w", newline='') as f:
        #     csvwriter = csv.writer(f)
        #     csvwriter.writerows(lines)

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            templine = []
            for item in line:
                item.decode('utf-8').encode('utf-8')
                templine.append(item)
            # line = item.decode('GB2312').encode('utf-8') for item in line
            lines.append(templine)
    return lines

def readTXT(filename):
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def tryFloat(value):
    try:
        value = float(value)
    except:
        value = value
    
    return value

def getColumn(lines, columnid, elementType=''):
    column = []
    for line in lines:
        try:
            value = line[columnid]
        except:
            continue
            
        if elementType == 'float':
            value = tryFloat(value)

        column.append(value)
    return column
