import os
import csvTools

bacteriapath = '/home/wangqiuli/Data/pneumonia/BacteriaScreened/'
funguspath = '/home/wangqiuli/Data/pneumonia/FungusScreened/'

bacFilelist = os.listdir(bacteriapath)
funFilelist = os.listdir(funguspath)

print(len(bacFilelist))
print(len(funFilelist))

baclines = []
funlines = []
for onebac in bacFilelist:
    baclines.append(str(onebac) + ',bacter')
for onefun in funFilelist:
    funlines.append(str(onefun) + ',fungus')

import random
random.shuffle(baclines)
random.shuffle(funlines)

alllines = baclines + funlines
random.shuffle(alllines)

csvTools.writeCSV('label/all.csv', alllines)
csvTools.writeCSV('label/bacter.csv', baclines)
csvTools.writeCSV('label/fungus.csv', funlines)

print('Done CSV!')