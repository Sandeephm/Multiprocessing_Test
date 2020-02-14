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
import concurrent.futures


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

if __name__ == '__main__':

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

    start_time = time.time()



    ecg_in = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0,12):
            ecg_in.append(executor.submit(denoising,data[:,i]))

        i = 0
        for f in concurrent.futures.as_completed(ecg_in):
            Denoised[:,i] = f.result()
            i = i + 1

    #for i in range(12):
    #        Denoised[:,i] = denoising(data[:,i])

    #print(time.process_time() - start)



    #boundary_I,R_peak_index_I = Boundary_Detection(Denoised[:,0])
    #boundary_II,R_peak_index_II = Boundary_Detection(Denoised[:,1])
    #boundary_III,R_peak_index_III = Boundary_Detection(Denoised[:,2])
    #boundary_avr,R_peak_index_avr = Boundary_Detection(Denoised[:,3])
    #boundary_avl,R_peak_index_avl = Boundary_Detection(Denoised[:,4])
    #boundary_avf,R_peak_index_avf = Boundary_Detection(Denoised[:,5])
    #boundary_V1,R_peak_index_V1 = Boundary_Detection(Denoised[:,6])
    #boundary_V2,R_peak_index_V2 = Boundary_Detection(Denoised[:,7])
    #boundary_V3,R_peak_index_V3 = Boundary_Detection(Denoised[:,8])
    #boundary_V4,R_peak_index_V4 = Boundary_Detection(Denoised[:,9])
    #boundary_V5,R_peak_index_V5 = Boundary_Detection(Denoised[:,10])
    #boundary_V6,R_peak_index_V6 = Boundary_Detection(Denoised[:,11])

    bd_param = []
    bd_results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0,12):
            bd_param.append(executor.submit(Boundary_Detection,Denoised[:,i]))

        for f in concurrent.futures.as_completed(bd_param):
            bd_results.append(f.result())
    time.sleep(0.5)
    print(bd_results[0][0])


    R_peak_idx = 0
    boundary_I = bd_results[0][0]
    R_peak_index_I = bd_results[0][1]


    for i in range(len(R_peak_index_I)):
            if( (boundary_I[0]<R_peak_index_I[i]) and (boundary_I[1]>R_peak_index_I[i]) ):
                    R_peak_idx = i
                    break

    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0,len(boundary_I) - 1):
                #P_wave,QRS_wave,T_wave = hybrid_tdmg_FE_new(Denoised[boundary_I[i] : boundary_I[i + 1],1],range(boundary_I[i],boundary_I[i + 1]), R_peak_index_I[R_peak_idx] - boundary_I[i])
                results.append(executor.submit(hybrid_tdmg_FE_new,Denoised[boundary_I[i] : boundary_I[i + 1],1],range(boundary_I[i],boundary_I[i + 1]), R_peak_index_I[R_peak_idx] - boundary_I[i]))
                R_peak_idx = R_peak_idx + 1

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    end_time = time.time()
    print(end_time-start_time)



'''
fe_param_I = np.zeros((1000 3),len(boundary_I) - 1)

for i in range(0,len(boundary_I) - 1):
        #P_wave,QRS_wave,T_wave = hybrid_tdmg_FE_new(Denoised[boundary_I[i] : boundary_I[i + 1],j],range(boundary_I[i],boundary_I[i + 1]), R_peak_index_I[R_peak_idx] - boundary_I[i])
        fe_param_I[i] = Denoised[:,1] + [boundary_I[i]] + [boundary_I[i+1]] +  [R_peak_index_I[R_peak_idx] - boundary_I[i]]
        R_peak_idx = R_peak_idx + 1


print(R_peak_index_I)

print(boundary_I)

print(fe_param_I[:,10000-10002])
'''
