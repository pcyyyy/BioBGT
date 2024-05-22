import numpy as np
from nilearn.connectome import ConnectivityMeasure
import os
import pandas as pd

df = pd.read_excel('C:\\Users\\Huang\\Downloads\\ADHD200_CC200_TCs_filtfix\\label.xlsx')
my_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
my_dict["KKI_1018959"] = 0

site = ["KKI", "NYU", "NeuroIMAGE", "OHSU"]
Path = "C:\\Users\\Huang\\Downloads\\ADHD200_CC200_TCs_filtfix\\KKI\\"

for root, directories, files in os.walk(Path):
    adj = []
    feature = []
    label = []
    count = 0
    for directory in directories:
        key = "KKI_" + directory.lstrip('0')
        D = Path + directory

        if my_dict.get(key) != None:
            for root2, directories2, files2 in os.walk(D):
                for file in files2:
                    if file[:2] == "sn" and "rest_1" in file:
                        openfile = D + "\\" + file
                        print(openfile)
                        with open(openfile,'r') as file:
                            data = []
                            for line in file.readlines():
                                if line[0] == 's':
                                    Temp = []
                                    for i in range(len(line.strip().split('\t'))):
                                        if i == 0 or i == 1:
                                            continue
                                        else:
                                            Temp.append(float(line.strip().split('\t')[i]))
                                    data.append(np.array(Temp))
                            time_series = np.stack(data)
                            #
                            # add feature alignment code here
                            #
                            correlation_measure = ConnectivityMeasure(kind='correlation')
                            correlation_matrix = correlation_measure.fit_transform([time_series])[0]

            adj.append(correlation_matrix)
            feature.append(np.transpose(time_series))
            count += int(my_dict[key])
            label.append(int(my_dict[key]))
