from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from Boundary_detection_Pyton_for_RPi import denoising,Boundary_Detection
from hybrid_tdmg_FE_new import hybrid_tdmg_FE_new
import os
import glob
import xlsxwriter
import time


files = glob.glob("E:\ECG Matlab\PTBDB_BDFE_GOLDEN\PTDB\*.csv")

'''
for filepath in files:

        list1 = []
        s = ""
        head_tail = os.path.split(filepath)
        tail = head_tail[1]
        tail= tail.split('.csv')
        name = tail[0]
        list1.append('E:\Python Updated Code\BD+FE_python\\')
        list1.append(name)
        list1.append('.xlsx')
        s = s.join(list1)
        print(s)


        data_f = pd.read_csv(filepath, sep=',',header=None)
        data = data_f.as_matrix()


        Denoised = np.zeros((len(data),12))


        for i in range(12):
                Denoised[:,i] = denoising(data[:,i])



        #X = np.arange(1,10001)

        #workbook = xlsxwriter.Workbook(s)
        #worksheet = workbook.add_worksheet()



        for j in range(12):

                boundary_I,R_peak_index_I = Boundary_Detection(Denoised[:,j])

                R_peak_idx = 0

                for i in range(len(R_peak_index_I)):
                        if( (boundary_I[0]<R_peak_index_I[i]) and (boundary_I[1]>R_peak_index_I[i]) ):
                                R_peak_idx = i
                                break

                P_array = []
                QRS_array = []
                T_array  = []

                start = time.process_time()

                for i in range(0,len(boundary_I) - 1):
                        P_wave,QRS_wave,T_wave = hybrid_tdmg_FE_new(Denoised[boundary_I[i] : boundary_I[i + 1],j],range(boundary_I[i],boundary_I[i + 1]), R_peak_index_I[R_peak_idx] - boundary_I[i])
                        P_array = P_array + P_wave
                        QRS_array = QRS_array + QRS_wave
                        T_array = T_array + T_wave
                        R_peak_idx = R_peak_idx + 1

                print(time.process_time() - start)
        '''
'''

        for j in range(12):
                plt.figure(j+1)

                bd_values = []
                for i in boundary_I:
                        bd_values.append(Denoised[i,j])

                R_values = []
                for i in R_peak_index_I:
                        R_values.append(Denoised[i,j])

                plt.plot(X,Denoised[:,j])
                plt.plot(boundary_I,bd_values,'o')
                plt.plot(R_peak_index_I,R_values,'o')


                row = 0
                column = j

                for item in boundary_I:
                        worksheet.write(row,column, item)
                        row += 1

                row = 0
                column = 11+j
                for item in R_peak_index_I:
                        worksheet.write(row,column, item)
                        row += 1


        #workbook.close()
        plt.show()
'''

filepath = files[0]
list1 = []
s = ""
head_tail = os.path.split(filepath)
tail = head_tail[1]
tail= tail.split('.csv')
name = tail[0]
list1.append('E:\Python Updated Code\BD+FE_python\\')
list1.append(name)
list1.append('.xlsx')
s = s.join(list1)
print(s)


data_f = pd.read_csv(filepath, sep=',',header=None)
data = data_f.as_matrix()


Denoised = np.zeros((len(data),12))


for i in range(12):
        Denoised[:,i] = denoising(data[:,i])

boundary_I,R_peak_index_I = Boundary_Detection(Denoised[:,1])

R_peak_idx = 0

for i in range(len(R_peak_index_I)):
        if( (boundary_I[0]<R_peak_index_I[i]) and (boundary_I[1]>R_peak_index_I[i]) ):
                R_peak_idx = i
                break

fe_param_I = np.zeros((1000 3),len(boundary_I) - 1)

for i in range(0,len(boundary_I) - 1):
        #P_wave,QRS_wave,T_wave = hybrid_tdmg_FE_new(Denoised[boundary_I[i] : boundary_I[i + 1],j],range(boundary_I[i],boundary_I[i + 1]), R_peak_index_I[R_peak_idx] - boundary_I[i])
        fe_param_I[i] = Denoised[:,1] + [boundary_I[i]] + [boundary_I[i+1]] +  [R_peak_index_I[R_peak_idx] - boundary_I[i]]
        R_peak_idx = R_peak_idx + 1


print(R_peak_index_I)

print(boundary_I)

print(fe_param_I[:,10000-10002])
