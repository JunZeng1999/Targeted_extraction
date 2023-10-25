import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from parse_data import get_EICs, get_EICs1, peak_detection1, cnn_data_preprocess0, cnn_data_preprocess, \
    cnn_data_preprocess1
import torch
from model.classifier_model import Classifier
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import os


class GUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.place()
        self.createWidget(master)

    def createWidget(self, master):
        self.f1 = tk.Canvas(master, width=280, height=210, bg='white')
        self.f2 = tk.Canvas(master, width=280, height=210, bg='white')
        self.f1.place(x=320, y=20)
        self.f2.place(x=320, y=260)
        self.f3 = tk.Frame(master, width=250, height=150, bg="#fff")
        self.f4 = tk.Frame(master, width=250, height=150, bg="#fff")
        self.f3.place(x=25, y=145)
        self.f4.place(x=25, y=320)
        self.l1 = tk.Label(master, text='EIC', font=('微软雅黑', 12),  width=8, height=2)
        self.l1.place(x=602, y=100)
        self.l2 = tk.Label(master, text='MS/MS', font=('微软雅黑', 12), width=8, height=2)
        self.l2.place(x=602, y=340)
        self.l3 = tk.Label(master, text='Blank', font=('微软雅黑', 10), width=10, height=2)
        self.l3.place(x=15, y=8)
        self.l4 = tk.Label(master, text='Admin.', font=('微软雅黑', 10), width=10, height=2)
        self.l4.place(x=15, y=38)
        self.l5 = tk.Label(master, text='m/z list', font=('微软雅黑', 10), width=10, height=2)
        self.l5.place(x=15, y=68)
        self.l6 = tk.Label(master, text='Result_existence_MS/MS', font=('微软雅黑', 9), width=30, height=1)
        self.l6.place(x=40, y=295)
        self.l7 = tk.Label(master, text='Result_inexistence_MS/MS', font=('微软雅黑', 9), width=30, height=1)
        self.l7.place(x=40, y=470)
        self.vv1 = tk.StringVar()
        self.e1 = tk.Entry(master, textvariable=self.vv1, width=25)
        self.e1.place(x=100, y=20)
        self.vv2 = tk.StringVar()
        self.e2 = tk.Entry(master, textvariable=self.vv2, width=25)
        self.e2.place(x=100, y=50)
        self.vv3 = tk.StringVar()
        self.e3 = tk.Entry(master, textvariable=self.vv3, width=25)
        self.e3.place(x=100, y=80)
        self.strvar = tk.StringVar()
        self.strvar.set('Mode')  # 设置默认值
        self.om = tk.OptionMenu(master, self.strvar, 'Positive', 'Negative')
        self.om['width'] = 10
        self.om.place(x=100, y=105)
        self.listbox1 = tk.Listbox(self.f3, width=35, height=8)
        self.listbox1.pack()
        self.listbox2 = tk.Listbox(self.f4, width=35, height=8)
        self.listbox2.pack()
        self.btn1 = tk.Button(master, text="→")
        self.btn1.place(x=280, y=200)
        self.btn1.bind("<Button-1>", self.Img_out1)
        self.btn2 = tk.Button(master, text="→")
        self.btn2.place(x=280, y=375)
        self.btn2.bind("<Button-1>", self.Img_out2)
        menubar = tk.Menu(master)
        menuFile = tk.Menu(menubar, tearoff=0)
        menuEdit = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Input", menu=menuFile)
        menubar.add_cascade(label="Output", menu=menuEdit)
        menuFile.add_command(label="Blank group.mzML", command=self.open_file)
        menuFile.add_command(label="Admin. group.mzML", command=self.open_file1)
        menuFile.add_command(label="m/z.csv", command=self.open_file2)
        menuEdit.add_command(label="Run", command=self.run_file)
        menuEdit.add_command(label="Export", command=self.export_file)
        master["menu"] = menubar

    def open_file(self, event=None):
        input_fiel = filedialog.askopenfilename(title='Import data for a blank group', initialdir='d:\\',
                                                filetype=[(".mzML", ".mzML")])
        if input_fiel:
            self.vv1.set(input_fiel)

    def open_file1(self, event=None):
        input_fiel = filedialog.askopenfilename(title='Import data for an administration group',
                                                initialdir='d:\\', filetype=[(".mzML", ".mzML")])
        if input_fiel:
            self.vv2.set(input_fiel)

    def open_file2(self, event=None):
        input_fiel = filedialog.askopenfilename(title='m/z list', initialdir='d:\\', filetype=[(".csv", ".csv")])
        if input_fiel:
            self.vv3.set(input_fiel)

    def Img_out1(self, event=None):
        global MS_MS_path1, EIC_path1, photo1, photo1_1, photo2, photo2_1
        out1_1 = self.listbox1.curselection()
        out1 = np.array(out1_1)
        if self.mode == 0:
            MS_MS_path1 = "MS/negative/photo/peak_ID_{:06}_MSMS.png".format(np.array(self.peak_ID)[out1[0]])
            EIC_path1 = "MS/negative/photo/peak_ID_{:06}.png".format(np.array(self.peak_ID)[out1[0]])
        if self.mode == 1:
            MS_MS_path1 = "MS/positive/photo/peak_ID_{:06}_MSMS.png".format(np.array(self.peak_ID)[out1[0]])
            EIC_path1 = "MS/positive/photo/peak_ID_{:06}.png".format(np.array(self.peak_ID)[out1[0]])
        photo1 = Image.open(MS_MS_path1)
        resized1 = photo1.resize((280, 210), Image.ANTIALIAS)
        photo1_1 = ImageTk.PhotoImage(resized1)
        self.f2.create_image(0, 0, image=photo1_1, anchor='nw')
        photo2 = Image.open(EIC_path1)
        resized2 = photo2.resize((280, 210), Image.ANTIALIAS)
        photo2_1 = ImageTk.PhotoImage(resized2)
        self.f1.create_image(0, 0, image=photo2_1, anchor='nw')

    def Img_out2(self, event=None):
        global EIC_path2, photo3, photo3_1
        out2_1 = self.listbox2.curselection()
        out2 = np.array(out2_1)
        if self.mode == 0:
            EIC_path2 = "MS/negative/photo_1/peak_ID_{:06}.png".format(np.array(self.peak_ID1)[out2[0]])
        if self.mode == 1:
            EIC_path2 = "MS/positive/photo_1/peak_ID_{:06}.png".format(np.array(self.peak_ID1)[out2[0]])
        photo3 = Image.open(EIC_path2)
        resized3 = photo3.resize((280, 210), Image.ANTIALIAS)
        photo3_1 = ImageTk.PhotoImage(resized3)
        self.f1.create_image(0, 0, image=photo3_1, anchor='nw')
        self.f2.delete('all')

    def run_file(self, event=None):
        global ms_path, w3, w4
        run_fiel1 = self.vv1.get()
        run_fiel2 = self.vv2.get()
        run_fiel3 = self.vv3.get()
        if run_fiel1 and run_fiel2 and run_fiel3:
            csv = pd.read_csv(run_fiel3)
            ms2_tolerate = 0.01
            mz_list = csv['m/z'].tolist()
            mz_list = np.array(mz_list)
            mode1 = self.strvar.get()
            if mode1 == 'Positive':
                self.mode = 1
                if not os.path.exists('./MS/positive'):
                    os.mkdir('./MS/positive')
                if not os.path.exists('./MS/positive/peak'):
                    os.mkdir('./MS/positive/peak')
                if not os.path.exists('./MS/positive/photo'):
                    os.mkdir('./MS/positive/photo')
                if not os.path.exists('./MS/positive/photo_1'):
                    os.mkdir('./MS/positive/photo_1')
                mz_list = mz_list + 1.007825
            if mode1 == 'Negative':
                self.mode = 0
                if not os.path.exists('./MS/negative'):
                    os.mkdir('./MS/negative')
                if not os.path.exists('./MS/negative/peak'):
                    os.mkdir('./MS/negative/peak')
                if not os.path.exists('./MS/negative/photo'):
                    os.mkdir('./MS/negative/photo')
                if not os.path.exists('./MS/negative/photo_1'):
                    os.mkdir('./MS/negative/photo_1')
                mz_list = mz_list - 1.007825
            mz_list = mz_list.tolist()
            eics, mz_value_1, scan_time_1, scan_time_2, scan_precursor_mz_2, scan_precursor_i_2, scan_mz_2, scan_i_2 = \
                get_EICs1(run_fiel2, mz_list, self.mode, delta_mz=0.01)
            peaks1, peaks, peak_widths_end1, peak_widths_end, peak_widths_end3, mz_end, cnn_data = \
                peak_detection1(eics, mz_value_1, intensity_threshold=100)
            peaks1_end2, peaks_end2, widths1_end2, widths_end2, widths3_end2, data_end2 = \
                cnn_data_preprocess1(peaks1, peaks, peak_widths_end1, peak_widths_end, peak_widths_end3, cnn_data)
            cnn_data_end = cnn_data_preprocess0(cnn_data)

            # get devices
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("using {} device.".format(device))

            # create model
            model_class = Classifier().to(device)

            # load weights
            weights_path = "./save_weights/Classifier99.pth"
            assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
            model_class.load_state_dict(torch.load(weights_path))
            model_class.to(device)

            self.peak_ID = []
            self.mz_out = []
            self.rt_min = []
            self.rt_max = []
            self.peak_height = []
            self.peak_ID1 = []
            self.mz_out1 = []
            self.rt_min1 = []
            self.rt_max1 = []
            self.peak_height1 = []
            aa = 1
            for i in range(len(cnn_data_end)):
                if len(cnn_data_end[i]) != 0:
                    scan_time_2 = np.array(scan_time_2)
                    scan_precursor_mz_2 = np.array(scan_precursor_mz_2)
                    scan_precursor_i_2 = np.array(scan_precursor_i_2)

                    # data process
                    if self.mode == 0:
                        ms_path = "./MS/negative/"
                    if self.mode == 1:
                        ms_path = "./MS/positive/"
                    x = np.arange(1, 129)
                    peaks1_end1 = peaks1_end2[i]
                    peaks_end1 = peaks_end2[i]
                    widths1_end1 = widths1_end2[i]
                    widths_end1 = widths_end2[i]
                    widths3_end1 = widths3_end2[i]
                    data_end1 = data_end2[i]
                    data_cnn = cnn_data_end[i]
                    data_mz = mz_end[i]
                    data_peak_point = peak_widths_end[i]
                    target_mz = mz_list[i]
                    for j in range(len(data_cnn)):
                        peaks1_end = peaks1_end1[j]
                        peaks_end = peaks_end1[j]
                        widths1_end = widths1_end1[j]
                        widths_end = widths_end1[j]
                        widths3_end = widths3_end1[j]
                        data_end = data_end1[j]
                        peaks1_end = np.array(peaks1_end)
                        peaks_end = np.array(peaks_end)
                        widths1_end = np.array(widths1_end)
                        widths_end = np.array(widths_end)
                        widths3_end = np.array(widths3_end)
                        data_end = np.array(data_end)

                        data_mz_1 = data_mz[j]
                        exist = (data_mz_1 > 0) * 1.0
                        mzmean = sum(data_mz_1) / sum(exist)

                        ab = np.arange(0, len(data_end))
                        aa += 1
                        data_cnn_1 = data_cnn[j]
                        data_cnn_1_1 = data_cnn_1.astype(np.float32)
                        data_cnn_1_1 = torch.from_numpy(data_cnn_1_1)
                        signal = data_cnn_1_1.reshape(1, 1, 128)
                        model_class.eval()
                        with torch.no_grad():
                            outputs = model_class(signal.to(device))
                            predict1 = torch.max(outputs, dim=1)[1]
                        if predict1 != 0:
                            plt.plot(ab, data_end)
                            plt.plot(peaks_end - widths3_end[0], data_end[peaks_end - widths3_end[0]], "x")
                            plt.plot(widths_end - widths3_end[0], data_end[widths_end - widths3_end[0]], "x")
                            plt.xlabel('scans number')
                            plt.ylabel('Intensity')
                            plt.title('mz = %.4f' % mzmean)
                            if self.mode == 0:
                                plt.savefig("MS/negative/peak/peak_ID_{:06}.png".format(aa))
                            if self.mode == 1:
                                plt.savefig("MS/positive/peak/peak_ID_{:06}.png".format(aa))
                            plt.clf()
                            aa += 1
                            point_1 = data_peak_point[j][0]
                            point_1_1 = data_peak_point[j][1]
                            time1_start = scan_time_1[point_1]
                            time1_end = scan_time_1[point_1_1]

                            EICs_blank, mz_value_blank, scan_time_blank = get_EICs(run_fiel1, target_mz, self.mode,
                                                                                   delta_mz=0.01)
                            scan_time_blank = np.array(scan_time_blank)
                            EICs_blank = np.array(EICs_blank)
                            time_start_blank = np.abs(scan_time_blank - time1_start)
                            closest1 = np.argmin(time_start_blank)
                            time_end_blank = np.abs(scan_time_blank - time1_end)
                            closest2 = np.argmin(time_end_blank)
                            width_blank = closest2 - closest1
                            w_left = int(max([closest1 - width_blank, 0]))
                            w_right = int(min([closest2 + width_blank, len(EICs_blank)]))
                            cnn_data_blank = EICs_blank[w_left:w_right]
                            cnn_data_end_blank = cnn_data_preprocess(cnn_data_blank)
                            if len(cnn_data_end_blank) > 0:
                                cnn_data_end_blank_1 = cnn_data_end_blank.astype(np.float32)
                                cnn_data_end_blank_1 = torch.from_numpy(cnn_data_end_blank_1)
                                signal_blank = cnn_data_end_blank_1.reshape(1, 1, 128)
                                model_class.eval()
                                with torch.no_grad():
                                    outputs_blank = model_class(signal_blank.to(device))
                                    outputs_blank_1 = torch.max(outputs_blank, dim=1)[1]

                                if outputs_blank_1 != 0:
                                    intensity_blank_max = max(EICs_blank[closest1:closest2])
                                    if intensity_blank_max == 0:
                                        intensity_blank_max = 1
                                else:
                                    intensity_blank_max = 1
                            else:
                                intensity_blank_max = 1

                            intensity = data_end[peaks_end - widths3_end[0]]
                            if intensity / intensity_blank_max >= 5:
                                mz_cha = abs(scan_precursor_mz_2 - target_mz)
                                mz_cha_new = np.where(abs(mz_cha) <= ms2_tolerate)[0]
                                intensity2 = []
                                scan_time_end2 = []
                                if len(mz_cha_new) > 0:
                                    for a in range(len(mz_cha_new)):
                                        scan_time2 = scan_time_2[mz_cha_new[a]]
                                        if time1_start < scan_time2 < time1_end:
                                            intensity2.append(scan_precursor_i_2[mz_cha_new[a]])
                                            scan_time_end2.append(scan_time2)
                                        else:
                                            intensity2.append(0)
                                            scan_time_end2.append(1000000)
                                    if len(intensity2) > 0 and max(intensity2) > 0:
                                        scan_time_end2 = np.array(scan_time_end2)
                                        scan_time_index = scan_time_1[peaks_end]
                                        scan_time_cha = np.abs(scan_time_end2 - scan_time_index)
                                        max_index = np.argmin(scan_time_cha)
                                        list1 = scan_mz_2[mz_cha_new[max_index]]
                                        list2 = scan_i_2[mz_cha_new[max_index]]
                                        name = "unknown_{:06}".format(aa - 1)
                                        full_path = ms_path + name + '.msp'
                                        w1 = 'NAME: ' + name
                                        w2 = 'PRECURSORMZ: %.4f' % (scan_precursor_mz_2[mz_cha_new[max_index]])
                                        if self.mode == 1:
                                            w3 = 'PRECURSORTYPE: [M+H]+'
                                            w4 = 'IONMODE: Positive'
                                        if self.mode == 0:
                                            w3 = 'PRECURSORTYPE: [M-H]-'
                                            w4 = 'IONMODE: Negative'
                                        w5 = 'RETENTIONTIME: %.2f' % (scan_time_2[mz_cha_new[max_index]])
                                        w6 = 'Num Peaks: ' + "{}".format(len(list1))
                                        doc = open(full_path, 'w')
                                        print(w1, file=doc)
                                        print(w2, file=doc)
                                        print(w3, file=doc)
                                        print(w4, file=doc)
                                        print(w5, file=doc)
                                        print(w6, file=doc)
                                        for kk in range(len(list1)):
                                            print(list1[kk], '\t', list2[kk], file=doc)
                                        doc.close()
                                        self.peak_ID.append(aa - 1)
                                        self.mz_out.append(mzmean)
                                        self.rt_min.append(time1_start)
                                        self.rt_max.append(time1_end)
                                        self.peak_height.append(intensity)
                                        list1_1 = np.array(list1)
                                        list2_1 = np.array(list2)
                                        plt.bar(list1_1, list2_1)
                                        plt.xlabel('m/z')
                                        plt.ylabel('Intensity')
                                        if self.mode == 0:
                                            plt.savefig("MS/negative/photo/peak_ID_{:06}_MSMS.png".format(aa - 1))
                                        if self.mode == 1:
                                            plt.savefig("MS/positive/photo/peak_ID_{:06}_MSMS.png".format(aa - 1))
                                        plt.clf()

                                        plt.plot(x, data_cnn_1)
                                        plt.xlabel('scans number')
                                        plt.ylabel('Intensity')
                                        plt.title('mz = %.4f  intensity = %.4f  rt = %.4f - %.4f' % (mzmean, intensity,
                                                                                                     time1_start,
                                                                                                     time1_end))
                                        if self.mode == 0:
                                            plt.savefig("MS/negative/photo/peak_ID_{:06}.png".format(aa - 1))
                                        if self.mode == 1:
                                            plt.savefig("MS/positive/photo/peak_ID_{:06}.png".format(aa - 1))
                                        plt.clf()
                                    else:
                                        self.peak_ID1.append(aa - 1)
                                        self.mz_out1.append(mzmean)
                                        self.rt_min1.append(time1_start)
                                        self.rt_max1.append(time1_end)
                                        self.peak_height1.append(intensity)
                                        plt.plot(x, data_cnn_1)
                                        plt.xlabel('scans number')
                                        plt.ylabel('Intensity')
                                        plt.title('mz = %.4f  intensity = %.4f  rt = %.4f - %.4f' % (mzmean, intensity,
                                                                                                     time1_start,
                                                                                                     time1_end))
                                        if self.mode == 0:
                                            plt.savefig("MS/negative/photo_1/peak_ID_{:06}.png".format(aa - 1))
                                        if self.mode == 1:
                                            plt.savefig("MS/positive/photo_1/peak_ID_{:06}.png".format(aa - 1))
                                        plt.clf()

            self.df = pd.DataFrame()
            self.df['peak_ID'] = self.peak_ID
            self.df['m/z'] = self.mz_out
            self.df['rtmin'] = self.rt_min
            self.df['rtmax'] = self.rt_max
            self.df['intensity'] = self.peak_height

            self.df1 = pd.DataFrame()
            self.df1['peak_ID'] = self.peak_ID1
            self.df1['m/z'] = self.mz_out1
            self.df1['rtmin'] = self.rt_min1
            self.df1['rtmax'] = self.rt_max1
            self.df1['intensity'] = self.peak_height1

            for i in range(len(self.peak_ID)):
                out_text1 = 'peak_ID =' + '%.d' % (self.peak_ID[i]) + ', ' + 'm/z = ' + '%.4f' % (self.mz_out[i]) + \
                            ', ' + 'rt_min = ' + '%.4f' % (self.rt_min[i]) + ', ' + 'rt_max = ' + '%.4f' % \
                            (self.rt_max[i]) + ', ' + 'peak_height = ' + '%.4f' % (self.peak_height[i])
                self.listbox1.insert(i, out_text1)

            for i in range(len(self.peak_ID1)):
                out_text2 = 'peak_ID =' + '%d' % (self.peak_ID1[i]) + ', ' + 'm/z = ' + '%.4f' % (self.mz_out1[i]) + \
                            ', ' + 'rt_min = ' + '%.4f' % (self.rt_min1[i]) + ', ' + 'rt_max = ' + '%.4f' % \
                            (self.rt_max1[i]) + ', ' + 'peak_height = ' + '%.4f' % (self.peak_height1[i])
                self.listbox2.insert(i, out_text2)

    def export_file(self, event=None):
        global out_path, out_path1
        folder_path = filedialog.askdirectory()
        if folder_path:
            if self.mode == 0:
                out_path = folder_path + '/Neg_result_existence_MSMS.csv'
                out_path1 = folder_path + '/Neg_result_inexistence_MSMS.csv'
            if self.mode == 1:
                out_path = folder_path + '/Pos_result_existence_MSMS.csv'
                out_path1 = folder_path + '/Pos_result_inexistence_MSMS.csv'
            self.df.to_csv(out_path)
            self.df1.to_csv(out_path1)


if __name__ == '__main__':
    window = tk.Tk()
    winWidth = 680
    winHeight = 500
    screenWidth = window.winfo_screenwidth()
    screenHeight = window.winfo_screenheight()
    x1 = int((screenWidth - winWidth) / 2)
    y1 = int((screenHeight - winHeight) / 2)
    window.title('Targeted extraction')
    window.geometry('%sx%s+%s+%s' % (winWidth, winHeight, x1, y1))
    window.resizable(0, 0)
    app = GUI(master=window)
    window.mainloop()