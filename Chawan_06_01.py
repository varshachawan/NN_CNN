# Chawan, Varsha Rani
# 1001-553-524
# 2018-12-10
# Assignment-06-01

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import Chawan_06_02
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn



class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Chawan Varsha Rani
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # set the properties of the row and columns in the master frame
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        self.master_frame.rowconfigure(2, weight=10, minsize=100, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        # Create an object for plotting graphs in the left frame
        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_error = CNN(self, self.left_frame, debug_print_flag=self.debug_print_flag)


class CNN:
    """
    This class creates and controls the sliders , buttons , drop down in the frame which
    are used to display decision boundary and generate samples and train .
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.alpha = 0.1
        self.lamda = 0.01
        self.height = 32
        self.width = 32
        self.channel = 3
        self.n_inputs = self.height * self.width
        self.F1 = 32
        self.K1 = 3
        self.conv1_stride = 1
        self.conv1_pad = "SAME"
        self.F2 = 32
        self.K2 = 3
        self.conv2_pad = "SAME"
        self.conv2_stride = 1
        self.F3 = 32
        self.K3 = 3
        self.conv3_stride = 1
        self.conv3_pad = "SAME"
        self.pool_map = self.F1
        self.pool2_map = self.F2
        self.pool3_map = self.F3
        self.n_fullyconnected = 32
        self.n_output = 10
        self.train_samples = 20
        self.total_samples = int(50000 * (self.train_samples / 100))
        self.x_train, self.y_train, one_hot_train = Chawan_06_02.load_training_data()
        self.x_train = self.x_train[:self.total_samples]

        self.y_train = self.y_train[:self.total_samples]

        self.x_test, self.y_test, one_hot_test = Chawan_06_02.load_test_data()

        self.epoch = 0
        self.epocs_list = []
        self.accu_list = []

        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name="X")
        self.Y = tf.placeholder(tf.int32, shape=[None])
        self.current_iterations = 0
        self.batch_size = 100
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamda)

        self.conv1 = tf.layers.conv2d(self.X, filters=self.F1, kernel_size=self.K1, strides=self.conv1_stride,
                                 padding=self.conv1_pad, activation=tf.nn.relu,kernel_regularizer=self.regularizer, name="conv1")
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        self.conv2 = tf.layers.conv2d(self.pool1, filters=self.F2, kernel_size=self.K2, strides=self.conv2_stride,
                                 padding=self.conv2_pad, activation=tf.nn.relu,kernel_regularizer=self.regularizer, name="conv2")
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        self.conv3 = tf.layers.conv2d(self.pool2, filters=self.F3, kernel_size=self.K3, strides=self.conv3_stride,
                                 padding=self.conv3_pad, activation=tf.nn.relu,kernel_regularizer=self.regularizer, name="conv3")
        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_list = self.pool3.get_shape().as_list()
        self.pool3_flat = tf.reshape(self.pool3, shape=[-1, shape_list[1] * shape_list[2] * shape_list[3]])
        self.fullyconn = tf.layers.dense(self.pool3_flat, self.n_fullyconnected, activation=None,kernel_regularizer=self.regularizer, name="fc1")
        self.logits = tf.layers.dense(self.fullyconn, self.n_output, name="output")
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure, self.axes_mul = plt.subplots(1, 2, figsize=(16, 6))

        self.axes = self.figure.gca()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the control widgets such as sliders ,buttons and dropdowns
        #########################################################################
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF", label="Learning Rate",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.lamda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF", label="Lambda",
                                     command=lambda event: self.lamda_slider_callback())
        self.lamda_slider.set(self.lamda)
        self.lamda_slider.bind("<ButtonRelease-1>", lambda event: self.lamda_slider_callback())
        self.lamda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.F1_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                     from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="F1",
                                     command=lambda event: self.F1_slider_callback())
        self.F1_slider.set(self.F1)
        self.F1_slider.bind("<ButtonRelease-1>", lambda event: self.F1_slider_callback())
        self.F1_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.F2_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                  from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="F2",
                                  command=lambda event: self.F2_slider_callback())
        self.F2_slider.set(self.F2)
        self.F2_slider.bind("<ButtonRelease-1>", lambda event: self.F2_slider_callback())
        self.F2_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.K1_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                            from_=3, to_=7, resolution=2, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF",
                                            label="K1",
                                            command=lambda event: self.K1_slider_callback())
        self.K1_slider.set(self.K1)
        self.K1_slider.bind("<ButtonRelease-1>", lambda event: self.K1_slider_callback())
        self.K1_slider.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.K2_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                  from_=3, to_=7, resolution=2, bg="#DDDDDD",
                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                  label="K2",
                                  command=lambda event: self.K2_slider_callback())
        self.K2_slider.set(self.K2)
        self.K2_slider.bind("<ButtonRelease-1>", lambda event: self.K2_slider_callback())
        self.K2_slider.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.train_samples_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                            from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF",
                                            label="Train Samples",
                                            command=lambda event: self.train_samples_slider_callback())
        self.train_samples_slider.set(self.train_samples)
        self.train_samples_slider.bind("<ButtonRelease-1>", lambda event: self.train_samples_slider_callback())
        self.train_samples_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)



        self.adjust_weight_button = tk.Button(self.controls_frame, text="Train", width=16,
                                              command=self.adjust_weight_button_callback)
        self.adjust_weight_button.grid(row=1, column=0)

        self.reset_weight_button = tk.Button(self.controls_frame, text="Reset Weights", width=16,
                                                 command=self.reset_weight_button_callback)
        self.reset_weight_button.grid(row=1, column=1)



    def plot_ErrorRate(self, epoc,acc,cm):

        #########################################################################
        #  Freeze the weights and calculate o/p , find the index and plot
        #########################################################################
        self.axes.cla()
        self.axes.xaxis.set_visible(True)

        self.axes_mul[0].cla()
        self.axes_mul[0].plot(epoc, acc, marker='o', markersize=5)
        self.axes_mul[0].set_title("CNN")
        self.axes_mul[0].set_xlabel('Epochs')
        self.axes_mul[0].set_ylabel('Percentage Error')


        self.axes_mul[1].cla()
        self.axes_mul[1].set_xlabel("Predicted")
        self.axes_mul[1].set_ylabel("True")
        plt.title("Confusion Matrix ")
        df_cm = pd.DataFrame(cm,range(10),range(10))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm,annot=True,cbar=False,fmt='g')


        plt.tight_layout()
        self.canvas.draw()


    def get_next_batch(self, batch_size):
        self.current_iterations = 0
        start_index = (self.current_iterations * batch_size) % len(self.y_train)
        end_index = start_index + batch_size
        x_batch = self.x_train[start_index:end_index]
        y_batch = self.y_train[start_index:end_index]
        self.current_iterations = self.current_iterations + 1
        return x_batch, y_batch

    def train_model(self):

        self.epoch += 1
        Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        loss = tf.reduce_mean(Xentropy)
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        optimizer = tf.train.GradientDescentOptimizer(self.alpha)
        # optimizer = tf.train.AdamOptimizer(self.alpha)
        training_op = optimizer.minimize(loss)

        num_exam = len(self.y_train)
        for iteration in range(num_exam // self.batch_size):
            X_batch, Y_batch = self.get_next_batch(self.batch_size)
            self.sess.run(training_op, feed_dict={self.X: X_batch, self.Y: Y_batch})
        self.test_data()


    def reset_tensor_variables(self):
         self.epoch = 0
         self.epocs_list = []
         self.accu_list = []


         tf.reset_default_graph()
         self.X = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name="X")
         self.Y = tf.placeholder(tf.int32, shape=[None])
         self.current_iterations = 0
         self.batch_size = 100
         self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamda)

         self.conv1 = tf.layers.conv2d(self.X, filters=self.F1, kernel_size=self.K1, strides=self.conv1_stride,
                                       padding=self.conv1_pad, activation=tf.nn.relu,
                                       kernel_regularizer=self.regularizer, name="conv1")
         self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
         self.conv2 = tf.layers.conv2d(self.pool1, filters=self.F2, kernel_size=self.K2, strides=self.conv2_stride,
                                       padding=self.conv2_pad, activation=tf.nn.relu,
                                       kernel_regularizer=self.regularizer, name="conv2")
         self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
         self.conv3 = tf.layers.conv2d(self.pool2, filters=self.F3, kernel_size=self.K3, strides=self.conv3_stride,
                                       padding=self.conv3_pad, activation=tf.nn.relu,
                                       kernel_regularizer=self.regularizer, name="conv3")
         self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
         shape_list = self.pool3.get_shape().as_list()
         self.pool3_flat = tf.reshape(self.pool3, shape=[-1, shape_list[1] * shape_list[2] * shape_list[3]])
         self.fullyconn = tf.layers.dense(self.pool3_flat, self.n_fullyconnected, activation=None,
                                          kernel_regularizer=self.regularizer, name="fc1")
         self.logits = tf.layers.dense(self.fullyconn, self.n_output, name="output")
         self.sess = tf.Session()
         self.init = tf.global_variables_initializer()
         self.sess.run(self.init)


    def test_data(self):
        correct = tf.nn.in_top_k(self.logits, self.Y, 1)

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracypercent = accuracy*100

        error = 100.00 - accuracypercent
        acc_test = error.eval(session=self.sess, feed_dict={self.X: self.x_test, self.Y: self.y_test})

        self.epocs_list.append(self.epoch)
        self.accu_list.append(acc_test)

        _, top_predicted_label = tf.nn.top_k(self.logits, k=1, sorted=False)

        top_predicted_label = tf.squeeze(top_predicted_label, axis=1)

        cm = tf.confusion_matrix(
            self.y_test, top_predicted_label, num_classes=self.n_output)
        cmeval = cm.eval(session=self.sess, feed_dict={self.X: self.x_test, self.Y: self.y_test})
        self.plot_ErrorRate(self.epocs_list, self.accu_list, cmeval)


    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())


    def lamda_slider_callback(self):
        self.lamda = np.float(self.lamda_slider.get())


    def F1_slider_callback(self):
        self.F1 = self.F1_slider.get()

        self.reset_tensor_variables()
        self.test_data()


    def F2_slider_callback(self):
        self.F2 = self.F2_slider.get()
        self.reset_tensor_variables()
        self.test_data()

    def K1_slider_callback(self):
        self.K1 = self.K1_slider.get()
        self.reset_tensor_variables()
        self.test_data()

    def K2_slider_callback(self):
        self.K2 = self.K2_slider.get()
        self.reset_tensor_variables()
        self.test_data()

    def train_samples_slider_callback(self):
        self.train_samples = self.train_samples_slider.get()
        self.total_samples = int(50000 * (self.train_samples / 100))
        self.x_train, self.y_train, one_hot_train = Chawan_06_02.load_training_data()
        self.x_train = self.x_train[:self.total_samples]
        self.y_train = self.y_train[:self.total_samples]

    def adjust_weight_button_callback(self): #train
        self.train_model()


    def reset_weight_button_callback(self):
        self.reset_tensor_variables()
        self.test_data()


        #########################################################################
        #  Logic to close Main Window
        #########################################################################

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()


main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_06 --  Chawan')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()