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

def Feature_extraction(ecg_signal,boundary,R_peak):

    R_peak_idx = 0

    for i in range(len(R_peak)):
            if( (boundary[0]<R_peak[i]) and (boundary[1]>R_peak[i]) ):
                    R_peak_idx = i
                    break

    P_vector = []
    QRS_vector = []
    T_vector = []

    for i in range(0,len(boundary) - 1):
        result = hybrid_tdmg_FE_new(ecg_signal[boundary[i] : boundary[i + 1]],range(boundary[i],boundary[i + 1]), R_peak[R_peak_idx] - boundary[i])
        R_peak_idx = R_peak_idx + 1
        P_vector = P_vector + result[0]
        QRS_vector = QRS_vector + result[1]
        T_vector = T_vector + result[2]

    return P_vector,QRS_vector,T_vector



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

    files = glob.glob("E:\ECG Matlab\PTBDB_BDFE_GOLDEN\PTDB\*.csv")

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

        start_time = time.time()

        ######################################## Filtering ECG Signal

        ecg_in = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0,12):
                ecg_in.append(executor.submit(denoising,data[:,i]))

            i = 0
            concurrent.futures.wait(ecg_in, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
            for f in ecg_in:
                Denoised[:,i] = f.result()
                i = i + 1


        ################################################################


        ######################################## Boundary Detection on ECG Signal
        bd_param = []
        bd_results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0,12):
                bd_param.append(executor.submit(Boundary_Detection,Denoised[:,i]))

            concurrent.futures.wait(bd_param, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
            for f in bd_param:
                bd_results.append(f.result())

        R_peak_idx = 0
        ################################################################

        fe_param = []
        fe_results = []
        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0,12):
                fe_param.append(executor.submit(Feature_extraction,Denoised[:,i],bd_results[i][0],bd_results[i][1]))

            concurrent.futures.wait(fe_param, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
            for f in fe_param:
                fe_results.append(f.result())
        '''

        for i in range(0,12):
            res = Feature_extraction(Denoised[:,i],bd_results[i][0],bd_results[i][1])
            fe_results.append(res)

        end_time = time.time()
        print(end_time-start_time)


        workbook = xlsxwriter.Workbook(s)
        worksheet = workbook.add_worksheet()

        for j in range(0,12):

            #plt.figure(j+1)

            X = np.arange(1,10001)

            bd_values = []
            for i in bd_results[j][0]:
                    bd_values.append(Denoised[i,j])

            R_values = []
            for i in bd_results[j][1]:
                    R_values.append(Denoised[i,j])


            P_wave_idx = []
            P_wave_values = []

            for i in fe_results[j][0]:
                if(i != -1):
                    i = int(i)
                    P_wave_idx.append(i)
                    P_wave_values.append(Denoised[i,j])

            row = 0
            column = j

            for item in P_wave_idx:
                    worksheet.write(row,column, item)
                    row += 1

            QRS_wave_idx = []
            QRS_wave_values = []
            for i in fe_results[j][1]:
                if(i != -1):

                    i = int(i)
                    QRS_wave_idx.append(i)
                    QRS_wave_values.append(Denoised[i,j])

            row = 0
            column = 12+j
            for item in QRS_wave_idx:
                    worksheet.write(row,column, item)
                    row += 1

            T_wave_idx = []
            T_wave_values = []
            for i in fe_results[j][2]:
                if(i != -1):
                    i = int(i)
                    T_wave_idx.append(i)
                    T_wave_values.append(Denoised[i,j])

            row = 0
            column = 24+j
            for item in T_wave_idx:
                    worksheet.write(row,column, item)
                    row += 1


            #plt.plot(X,Denoised[:,j])
            #plt.plot(bd_results[j][0],bd_values,'o')
            #plt.plot(bd_results[j][1],R_values,'o')
            #plt.plot(P_wave_idx,P_wave_values,'*')
            #plt.plot(QRS_wave_idx,QRS_wave_values,'*')
            #plt.plot(T_wave_idx,T_wave_values,'*')

        workbook.close()
