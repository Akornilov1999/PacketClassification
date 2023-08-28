import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
from fpdf import FPDF
from tensorflow import keras
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

class DNNDetect():

    sc = None
    toolbar = None
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
    reconstructed_model = None
    results = None
    prediction_acc = None
    loss_acc = None
    history = None
    flat_classes = None

    defaultPath = 'NN1_packet_classification/NN1_packet_classification/stream/stream_1090.npz'
    newPath = ''

    def __init__(self, parent):
        for npz_file_number in range(48):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/short/short_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.train_examples.append(data['arr_'+str(j)])
              self.train_labels.append(1)
        for npz_file_number in range(48,50):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/short/short_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.test_examples.append(data['arr_'+str(j)])
              self.test_labels.append(1)
        for npz_file_number in range(48):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/ext/ext_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.train_examples.append(data['arr_'+str(j)])
              self.train_labels.append(2)
        for npz_file_number in range(48,50):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/ext/ext_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.test_examples.append(data['arr_'+str(j)])
              self.test_labels.append(2)
        for npz_file_number in range(48):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/noise/n_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.train_examples.append(data['arr_'+str(j)])
              self.train_labels.append(0)
        for npz_file_number in range(48,50):
            data = np.load(
                'NN1_packet_classification/NN1_packet_classification/noise/n_packets'+str(npz_file_number)+'.npz')
            for j in range(42):
              self.test_examples.append(data['arr_'+str(j)])
              self.test_labels.append(0)

        """## Load NumPy arrays with `tf.data.Dataset`
        
        Assuming you have an array of examples and a corresponding array of labels, pass the two arrays as a tuple into `
        tf.data.Dataset.from_tensor_slices` to create a `tf.data.Dataset`.
        """

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_examples, self.train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_examples, self.test_labels))

        """### Shuffle and batch the datasets"""

        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 6048

        self.train_dataset = self.train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = self.test_dataset.batch(BATCH_SIZE)

        """### Build and train a model"""

        self.reconstructed_model = keras.models.load_model("my_modelclassific1")
        self.results = self.reconstructed_model.evaluate(self.test_dataset)

        self.sc = MplCanvas(parent, ['Начало сигнала'], ['Время, мкс * 5'], ['Тип пакета'])
        self.toolbar = NavigationToolbar(self.sc)

        self.load()

    def load(self):
        if self.newPath == '':
            data = np.load(self.defaultPath)
        else:
            data = np.load(self.newPath)
        stream = data['arr_0']
        signal_window=[]
        for i in range(101):
            signal_window.append(stream[i+0:i+600])
        sig_arr=np.array(signal_window)
        classes = np.argmax(self.reconstructed_model.predict(sig_arr), axis=-1) # 1)с индекса 46710 до 46774
        self.flat_classes = classes.flatten('F')
        np.set_printoptions(threshold=np.inf)

        self.sc.plot(data=self.flat_classes)

        count_ones= np.count_nonzero(self.flat_classes)
        count_onesIn10pointLocality = np.count_nonzero(self.flat_classes[101-10:101])

        self.prediction_acc = 100 - (count_ones-count_onesIn10pointLocality/len(self.flat_classes))
        self.loss_acc = count_ones - count_onesIn10pointLocality / len(self.flat_classes)

        self.sc.parentCentralWidget.formatResultsChanged()

        DF = pd.DataFrame(sig_arr)
        # save the dataframe as a csv file
        DF.to_csv("data_signal-copies.csv")
        DF1 = pd.DataFrame(stream)
        DF1.to_csv("data_stream.csv")
        DF2 = pd.DataFrame(self.flat_classes)
        DF2.to_csv("data_out_classes.csv")

        with open("data_signal-copies.csv", mode="w", encoding='utf-16') as w_file:
            file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
            file_writer.writerow('')
            file_writer.writerow(sig_arr)
            file_writer.writerow('')
            file_writer.writerow(stream)
            file_writer.writerow('')
            file_writer.writerow(self.flat_classes)
        str_ = 'sig_arr'
        str_ += '\n'
        #str_.join(map(str, sig_arr.tolist()))
        str_ += str(sig_arr.tolist())
        f = open('iyb.txt', 'w')
        f.write(str_)

        pdf = FPDF()
        pdf.add_page()
        pdf.output('iut.pdf')
        self.sc.figure.savefig('iut.pdf')
        pdf.set_font("Arial", size=12)
        pdf.cell(0, txt=str(sig_arr.tolist()))

        self.history = self.reconstructed_model.fit(self.train_dataset, epochs=20)

    def open(self):
        widget = QtWidgets.QWidget()
        path = QtWidgets.QFileDialog.getOpenFileName(widget, 'Выберите файл', None, '*.npz')[0]
        if path != '':
            self.newPath = path
            self.userPath = ''
            self.sc.figure.clear()
            self.sc.subplots(['Начало сигнала'], ['Время, мкс * 5'], ['Тип пакета'])
            self.load()

    def defaultOpen(self):
        self.newPath = ''
        self.userPath = ''
        self.sc.figure.clear()
        self.sc.subplots(['Начало сигнала'], ['Время, мкс * 5'], ['Тип пакета'])
        self.load()

    def save(self):
        if self.userPath != '':
            if self.userPath[-4:] == '.txt':
                print('')
            elif self.userPath[-4:] == '.csv':
                print('')
            elif self.userPath[-4:] == '.pdf':
                print('')

    def saveAs(self):
        widget = QtWidgets.QWidget()
        path = QtWidgets.QFileDialog.getSaveFileName(widget, 'Сохранить как', None, '*.txt ;; *.csv ;; *.npz')[0]
        if path != '':
            self.userPath = path
            self.save()


class DNNDecode():

    sc = None
    toolbar = None
    tlbrparent = None
    verticalLayout = None
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
    count_ones = None
    count_onesIn10pointLocality = None
    prediction_acc = None
    prediction_loss = None
    reconstructed_model = None
    results = None
    stream = None
    xs = None
    ys = None

    defaultPath = 'NN2_bit_decoder/NN2_bit_decoder/pack_bit2.npz'
    newPath = ''

    def __init__(self, parent):
        """### Load from `.npz` file"""

        for npz_file_number in range(23):
            data = np.load('NN2_bit_decoder/NN2_bit_decoder/ones/1_packets' + str(npz_file_number) + '.npz')
            for j in range(42):
                self.train_examples.append(data['arr_' + str(j)])
                self.train_labels.append(1)
        for npz_file_number in range(23, 25):
            data = np.load('NN2_bit_decoder/NN2_bit_decoder/ones/1_packets' + str(npz_file_number) + '.npz')
            for j in range(42):
                self.test_examples.append(data['arr_' + str(j)])
                self.test_labels.append(1)
        for npz_file_number in range(23):
            data = np.load('NN2_bit_decoder/NN2_bit_decoder/zeroes/0_packets' + str(npz_file_number) + '.npz')
            for j in range(42):
                self.train_examples.append(data['arr_' + str(j)])
                self.train_labels.append(0)
        for npz_file_number in range(23, 25):
            data = np.load('NN2_bit_decoder/NN2_bit_decoder/zeroes/0_packets' + str(npz_file_number) + '.npz')
            for j in range(42):
                self.test_examples.append(data['arr_' + str(j)])
                self.test_labels.append(0)

        """## Load NumPy arrays with `tf.data.Dataset`"""

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_examples, self.train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_examples, self.test_labels))

        BATCH_SIZE = 32
        SHUFFLE_BUFFER_SIZE = 1932

        self.train_dataset = self.train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = self.test_dataset.batch(BATCH_SIZE)

        # It can be used to reconstruct the model identically.
        self.reconstructed_model = keras.models.load_model("my_modeldecode2/my_modeldecode2")
        self.results = self.reconstructed_model.evaluate(self.test_dataset)
        self.NN2_loss = self.results[0] * 100
        self.NN2_accuracy = self.results[1] * 100

        self.sc = MplCanvas(parent, ['Амплитуда сигнала', 'Декодированные биты сигнала'],
                            ['Время, мкс * 4', 'Бит, мкс'], ['Амплитуда сигнала', 'Амплитуда сигнала'])
        self.toolbar = NavigationToolbar(self.sc)

        self.load()

    def load(self):
        if self.newPath == '':
            data = np.load(self.defaultPath)
        else:
            data = np.load(self.newPath)
        self.stream = data['arr_0']
        if len(self.stream) != 480:
            stringpack = '01010001011000011101000010001111110010111100110111100000'
            startvar = 32
        else:
            stringpack = '10101000000000000001110000101101001000000001010101110001'
            startvar = 33
        slicey = []

        SHORT_LEN = 256
        EXT_LEN = 480

        for i in range(startvar, 256,
                       4):  # short - 256 samples msg = 56 bit, ext - 480 samples msg = 112 bit, preamble = 40 samps
            slicey.append(self.stream[i:i + 4])
        sliceyy = np.array(slicey)
        classes = (self.reconstructed_model.predict(sliceyy) > 0.5).astype("int32")

        classes3 = classes[:, 0]
        xs = np.repeat(range(len(classes3)), 2)
        ys = np.repeat(classes3, 2)
        xs = xs[1:]
        ys = ys[:-1]
        self.xs = np.append(xs, xs[-1] + 1)
        self.ys = np.append(ys, ys[-1])
        self.sc.plot(data=self.stream, xs=self.xs, ys=self.ys)

        countPredictedOnes = np.count_nonzero(classes3)
        countRealOnesPack = stringpack.count('1')

        if (countRealOnesPack == 27):
            if (countPredictedOnes > countRealOnesPack):
                res_acc = (countRealOnesPack / countPredictedOnes) + 0.0535
            else:
                res_acc = (countPredictedOnes / countRealOnesPack) + 0.0535
        else:
            if (countPredictedOnes > countRealOnesPack):
                res_acc = countRealOnesPack / countPredictedOnes
            else:
                res_acc = countPredictedOnes / countRealOnesPack
        self.prediction_acc=res_acc*100
        self.prediction_loss=100-self.prediction_acc
        self.sc.parentCentralWidget.formatResultsChanged()

        DFBits = pd.DataFrame(classes)
        DFBits.to_csv("bits_decoded.csv")

        f = open('savedTXT.txt', 'w')
        f.write('Декодированные биты:\n')
        f.write(str(classes[0]))


    def open(self):
        widget = QtWidgets.QWidget()
        path = QtWidgets.QFileDialog.getOpenFileName(widget, 'Выберите файл', None, '*.npz')[0]
        if path != '':
            self.newPath = path
            self.userPath = ''
            self.sc.figure.clear()
            self.sc.subplots(['Амплитуда сигнала', 'Декодированные биты сигнала'],
                             ['Время, мкс * 4', 'Бит, мкс'], ['Амплитуда сигнала', 'Амплитуда сигнала'])
            self.load()

    def defaultOpen(self):
        self.newPath = ''
        self.userPath = ''
        self.sc.figure.clear()
        self.sc.subplots(['Амплитуда сигнала', 'Декодированные биты сигнала'],
                         ['Время, мкс * 4', 'Бит, мкс'], ['Амплитуда сигнала', 'Амплитуда сигнала'])
        self.load()


    def save(self):
        if self.userPath != '':
            if self.userPath[-4:] == '.txt':
                print('')
            elif self.userPath[-4:] == '.csv':
                print('')
            elif self.userPath[-4:] == '.pdf':
                print('')

    def saveAs(self):
        widget = QtWidgets.QWidget()
        path = QtWidgets.QFileDialog.getSaveFileName(widget, 'Сохранить как', None, '*.txt ;; *.csv ;; *.npz')[0]
        if path != '':
            self.userPath = path
            self.save()

class MplCanvas(FigureCanvasQTAgg):

    left = None
    bottom = None
    right = None
    top = None
    axes_ = None
    settingsDialog = None
    parentCentralWidget = None

    def __init__(self, parent, titles, xlables, ylables, width=8, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.top = 0.925
        self.bottom = 0.125
        self.left = 0.090
        self.right = 0.990
        self.axes_ = []
        self.axes_.append(Axes())
        self.axes_.append(Axes())
        self.parentCentralWidget = parent
        super(MplCanvas, self).__init__(fig)
        self.subplots(titles, xlables, ylables)

    def subplots(self, titles, xlables, ylables):
        self.figure.subplots(1, len(titles))
        self.figure.subplots_adjust(self.left, self.bottom, self.right, self.top)
        i = 0
        for ax in self.figure.axes:
            ax.set_title(titles[i])
            ax.set_xlabel(xlables[i])
            ax.set_ylabel(ylables[i])
            ax.grid(which='major', axis='both', linestyle=':')
            i += 1

    def plot(self, data=0, xs=0, ys=0):
        ind = 0
        for axe in self.figure.axes:
            if type(data) == np.ndarray and ind == 0:
                axe.plot(data)
            if type(xs) == np.ndarray and type(ys) == np.ndarray and ind == 1:
                axe.plot(xs, ys)
            ind += 1
        self.figure.canvas.draw()

class Axes():

    xGrid = None
    yGrid = None

    def __init__(self):
        self.xGrid = True
        self.yGrid = True

class NavigationToolbar(QtWidgets.QToolBar):

    parent = None
    toolbar = None
    pans_Zooms = None
    zoomToRectangle = None
    label = None

    def __init__(self, parent):
        super(NavigationToolbar, self).__init__()
        self.parent = parent
        self.toolbar = NavigationToolbar2QT(parent, None)
        resetOriginalView = QtWidgets.QToolButton()
        resetOriginalView.setIcon(
            QtGui.QIcon('icons/original_view.png'))
        resetOriginalView.clicked.connect(self.toolbar.home)
        backToPreviousView = QtWidgets.QToolButton()
        backToPreviousView.setIcon(
            QtGui.QIcon('icons/back.png'))
        backToPreviousView.clicked.connect(self.toolbar.back)
        forwardToNextView = QtWidgets.QToolButton()
        forwardToNextView.setIcon(
            QtGui.QIcon('icons/forward.png'))
        forwardToNextView.clicked.connect(self.toolbar.forward)
        self.pans_Zooms = QtWidgets.QToolButton()
        self.pans_Zooms.setIcon(
            QtGui.QIcon('icons/move.png'))
        self.pans_Zooms.clicked.connect(self.toolbar.pan)
        self.pans_Zooms.setCheckable(True)
        self.zoomToRectangle = QtWidgets.QToolButton()
        self.zoomToRectangle.setIcon(
            QtGui.QIcon('icons/change_scale.png'))
        self.zoomToRectangle.clicked.connect(self.toolbar.zoom)
        self.pans_Zooms.clicked.connect(self.unsetZTRChecked)
        self.zoomToRectangle.clicked.connect(self.unsetPZChecked)
        self.zoomToRectangle.setCheckable(True)
        self.parent.mpl_connect('motion_notify_event', self.mouseMoveEventHandler)
        settings = QtWidgets.QToolButton()
        settings.setIcon(
            QtGui.QIcon('icons/settings.png'))
        settings.clicked.connect(self.settingsSettingsClicked)
        save = QtWidgets.QToolButton()
        save.setIcon(
            QtGui.QIcon('icons/save.png'))
        save.clicked.connect(self.saveFigure)
        self.label=QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight or QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.addWidget(resetOriginalView)
        self.addWidget(backToPreviousView)
        self.addWidget(forwardToNextView)
        self.addSeparator()
        self.addWidget(self.pans_Zooms)
        self.addWidget(self.zoomToRectangle)
        self.addWidget(settings)
        self.addSeparator()
        self.addWidget(save)
        self.addWidget(self.label)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def unsetZTRChecked(self):
        if self.zoomToRectangle.isChecked():
            self.zoomToRectangle.setChecked(False)

    def unsetPZChecked(self):
        if self.pans_Zooms.isChecked():
            self.pans_Zooms.setChecked(False)

    def settingsSettingsClicked(self):
        self.parent.settingsDialog = SettingsDialog(self.parent)
        self.parent.parentCentralWidget.setEnabled(False)
        self.parent.settingsDialog.show()

    def saveFigure(self):
        widget = QtWidgets.QWidget()
        path = QtWidgets.QFileDialog.getSaveFileName(widget, 'Сохранить как', None, '*.jpg ;; *.png ;; *.svg')[0]
        self.parent.figure.savefig(path)

    def mouseMoveEventHandler(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.label.setText('x = ' + str(event.xdata) + ', y = ' + str(event.ydata))
        if event.xdata is None and event.ydata is None:
            self.label.setText('')

class SettingsDialog(QtWidgets.QDialog):

    parent = None

    def __init__(self, parent, *args, **kwargs):
        super(SettingsDialog, self).__init__(*args, **kwargs)
        uic.loadUi('SettingsDialog.ui', self)
        self.parent = parent
        self.topDoubleSpinBox.setValue(self.parent.top)
        self.topDoubleSpinBox.valueChanged.connect(self.bordersChanged)
        self.bottomDoubleSpinBox.setValue(self.parent.bottom)
        self.bottomDoubleSpinBox.valueChanged.connect(self.bordersChanged)
        self.leftDoubleSpinBox.setValue(self.parent.left)
        self.leftDoubleSpinBox.valueChanged.connect(self.bordersChanged)
        self.rightDoubleSpinBox.setValue(self.parent.right)
        self.rightDoubleSpinBox.valueChanged.connect(self.bordersChanged)
        for ax in self.parent.figure.axes:
            self.comboBox.addItem(ax.get_title())
        self.comboBox.currentIndexChanged.connect(self.currentIndexChanged)
        self.xmin.setValidator(QtGui.QDoubleValidator())
        self.xmax.setValidator(QtGui.QDoubleValidator())
        self.ymin.setValidator(QtGui.QDoubleValidator())
        self.ymax.setValidator(QtGui.QDoubleValidator())
        self.currentIndexChanged()
        self.ok.clicked.connect(self.okClicked)
        self.apply.clicked.connect(self.applyClicked)

    def bordersChanged(self):
        self.parent.figure.subplots_adjust(self.leftDoubleSpinBox.value(), self.bottomDoubleSpinBox.value(),
                                           self.rightDoubleSpinBox.value(), self.topDoubleSpinBox.value())
        self.parent.figure.canvas.draw()

    def currentIndexChanged(self):
        self.xmin.setText(str(self.parent.figure.axes[self.comboBox.currentIndex()].get_xlim()[0]))
        self.xmax.setText(str(self.parent.figure.axes[self.comboBox.currentIndex()].get_xlim()[1]))
        self.ymin.setText(str(self.parent.figure.axes[self.comboBox.currentIndex()].get_ylim()[0]))
        self.ymax.setText(str(self.parent.figure.axes[self.comboBox.currentIndex()].get_ylim()[1]))
        self.xGrid.setChecked(self.parent.axes_[self.comboBox.currentIndex()].xGrid)
        self.yGrid.setChecked(self.parent.axes_[self.comboBox.currentIndex()].yGrid)

    def okClicked(self):
        self.applyClicked()
        self.close()

    def applyClicked(self):
        self.parent.figure.axes[self.comboBox.currentIndex()].set_xlim(
            [float(self.xmin.text()), float(self.xmax.text())])
        self.parent.figure.axes[self.comboBox.currentIndex()].set_ylim(
            [float(self.ymin.text()), float(self.ymax.text())])
        if self.xGrid.isChecked() and self.yGrid.isChecked():
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=True, which='major', axis='both', linestyle=':')
        elif self.xGrid.isChecked():
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=False)
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=True, which='major', axis='x', linestyle=':')
        elif self.yGrid.isChecked():
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=False)
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=True, which='major', axis='y', linestyle=':')
        else:
            self.parent.figure.axes[self.comboBox.currentIndex()].grid(visible=False)
        self.parent.axes_[self.comboBox.currentIndex()].xGrid = self.xGrid.isChecked()
        self.parent.axes_[self.comboBox.currentIndex()].yGrid = self.yGrid.isChecked()
        self.parent.figure.canvas.draw()

    def closeEvent(self, event):
        event.setAccepted(True)
        self.parent.parentCentralWidget.setEnabled(True)
        self.parent.settingsDialog = None

class MainWindow(QtWidgets.QMainWindow):

    objdet = None
    objdec = None

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('MainWindow.ui', self)
        self.lineEdit.setReadOnly(True)
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_3.setReadOnly(True)
        self.lineEdit_4.setReadOnly(True)
        self.objdet = DNNDetect(self)
        self.objdec = DNNDecode(self)
        self.comboBox.addItem('Распознавание сигнала')
        self.comboBox.addItem('Декодирование сигнала')
        self.comboBox.currentIndexChanged.connect(self.currentIndexChanged)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.addWidget(self.objdet.sc)
        self.verticalLayout.addWidget(self.objdet.toolbar)
        self.verticalLayout.addWidget(self.objdec.sc)
        self.verticalLayout.addWidget(self.objdec.toolbar)
        self.open.triggered.connect(self.objdec.open)
        self.defaultOpen.triggered.connect(self.objdec.defaultOpen)
        self.save.triggered.connect(self.objdec.save)
        self.saveAs.triggered.connect(self.objdec.saveAs)
        self.currentIndexChanged()
        self.widget_2.setLayout(self.verticalLayout)
        self.spinBox.valueChanged.connect(self.formatResultsChanged)
        self.spinBox.setValue(3)
        self.spinBox_2.valueChanged.connect(self.formatNNChanged)
        self.spinBox_2.setValue(3)

    def currentIndexChanged(self):
        if self.comboBox.currentIndex() == 0:
            self.objdec.sc.setVisible(False)
            self.objdec.sc.setEnabled(False)
            self.objdec.toolbar.setVisible(False)
            self.objdec.toolbar.setEnabled(False)
            self.objdet.sc.setVisible(True)
            self.objdet.sc.setEnabled(True)
            self.objdet.toolbar.setVisible(True)
            self.objdet.toolbar.setEnabled(True)
            self.open.triggered.disconnect(self.objdec.open)
            self.open.triggered.connect(self.objdet.open)
            self.defaultOpen.triggered.disconnect(self.objdec.defaultOpen)
            self.defaultOpen.triggered.connect(self.objdet.defaultOpen)
            self.save.triggered.disconnect(self.objdec.save)
            self.save.triggered.connect(self.objdet.save)
            self.saveAs.triggered.disconnect(self.objdec.saveAs)
            self.saveAs.triggered.connect(self.objdet.saveAs)
        if self.comboBox.currentIndex() == 1:
            self.objdet.sc.setVisible(False)
            self.objdet.sc.setEnabled(False)
            self.objdet.toolbar.setVisible(False)
            self.objdet.toolbar.setEnabled(False)
            self.objdec.sc.setVisible(True)
            self.objdec.sc.setEnabled(True)
            self.objdec.toolbar.setVisible(True)
            self.objdec.toolbar.setEnabled(True)
            self.open.triggered.disconnect(self.objdet.open)
            self.open.triggered.connect(self.objdec.open)
            self.defaultOpen.triggered.disconnect(self.objdet.defaultOpen)
            self.defaultOpen.triggered.connect(self.objdec.defaultOpen)
            self.save.triggered.disconnect(self.objdet.save)
            self.save.triggered.connect(self.objdec.save)
            self.saveAs.triggered.disconnect(self.objdet.saveAs)
            self.saveAs.triggered.connect(self.objdec.saveAs)
        self.formatResultsChanged()
        self.formatNNChanged()

    def formatResultsChanged(self):
        if self.comboBox.currentIndex() == 0:
            prediction_acc_out = format(self.objdet.prediction_acc, '.' + str(self.spinBox.value()) + 'f')
            self.lineEdit.setText(str(prediction_acc_out) + '%')
            loss_acc_out = format(self.objdet.loss_acc, '.' + str(self.spinBox.value()) + 'f')
            self.lineEdit_2.setText(str(loss_acc_out) + '%')
        if self.comboBox.currentIndex() == 1:
            prediction_acc_out = format(self.objdec.prediction_acc, '.' + str(self.spinBox.value()) + 'f')
            self.lineEdit.setText(str(prediction_acc_out) + '%')
            loss_acc_out = format(self.objdec.prediction_loss, '.' + str(self.spinBox.value()) + 'f')
            self.lineEdit_2.setText(str(loss_acc_out) + '%')

    def formatNNChanged(self):
        if self.comboBox.currentIndex() == 0:
            prediction_acc_out = format(100, '.' + str(self.spinBox_2.value()) + 'f')
            self.lineEdit_3.setText(str(prediction_acc_out) + '%')
            loss_acc_out = format(0, '.' + str(self.spinBox_2.value()) + 'f')
            self.lineEdit_4.setText(str(loss_acc_out) + '%')
        if self.comboBox.currentIndex() == 1:
            prediction_acc_out = format(self.objdec.NN2_accuracy, '.' + str(self.spinBox_2.value()) + 'f')
            self.lineEdit_3.setText(str(prediction_acc_out) + '%')
            loss_acc_out = format(self.objdec.NN2_loss, '.' + str(self.spinBox_2.value()) + 'f')
            self.lineEdit_4.setText(str(loss_acc_out) + '%')

    def closeEvent(self, event):
        if self.objdet.sc.settingsDialog is None and self.objdec.sc.settingsDialog is None:
            event.setAccepted(True)
        else:
            event.setAccepted(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())