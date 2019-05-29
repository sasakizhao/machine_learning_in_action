import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import Ch09.my_reg_tree as reg_tree


def re_draw(tol_s, tol_n):
    re_draw.f.clf()
    re_draw.ax = re_draw.f.add_subplot(111)
    if chk_btn_var.get():
        if tol_n < 2:
            tol_n = 2
        my_tree = reg_tree.create_tree(re_draw.raw_dat, reg_tree.model_leaf, \
                                       reg_tree.model_err, ops=(tol_s, tol_n))
        y_hat = reg_tree.create_fore_cast(my_tree, re_draw.test_dat, reg_tree.model_tree_eval)
    else:
        my_tree = reg_tree.create_tree(re_draw.raw_dat, ops=(tol_s, tol_n))
        y_hat = reg_tree.create_fore_cast(my_tree, re_draw.test_dat)
    re_draw.ax.scatter(re_draw.raw_dat[:, 0].A.tolist(), re_draw.raw_dat[:, 1].A.tolist(), color='r', s=5)
    re_draw.ax.plot(re_draw.test_dat, y_hat, linewidth=2.0)
    re_draw.canvas.draw()


def draw_new_tree():
    tol_n, tol_s = get_inputs()
    re_draw(tol_n, tol_s)



def get_inputs():
    try:
        tol_n = int(tol_n_entry.get())
    except:
        tol_n = 10
        print("tol_n")
        tol_n_entry.delete(0)
        tol_n_entry.insert(0, '10')
    try:
        tol_s = float(tol_s_entry.get())
    except:
        tol_s = 1.0
        print("tol_n")
        tol_s_entry.delete(0)
        tol_s_entry.insert(0, '1.0')
    return tol_n, tol_s



root = tk.Tk()

re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tk.Label(root, text='tol_n').grid(row=1, column=0)
tol_n_entry = tk.Entry(root)
tol_n_entry.grid(row=1, columnspan=1)
tol_n_entry.insert(0, '10')

tk.Label(root, text='tol_s').grid(row=2, column=0)
tol_s_entry = tk.Entry(root)
tol_s_entry.grid(row=2, columnspan=1)
tol_s_entry.insert(0, '1.0')

tk.Button(root, text='ReDraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

chk_btn_var = tk.IntVar()
cnk_btn = tk.Checkbutton(root, text='Model Tree', variable=chk_btn_var)
cnk_btn.grid(row=3, column=0, columnspan=2)

re_draw.raw_dat = np.mat(reg_tree.load_dataset('sine.txt'), dtype=np.float)
re_draw.test_dat = np.arange(min(re_draw.raw_dat[:, 0]), max(re_draw.raw_dat[:, 0]), 0.01)

re_draw(1.0, 10)
root.mainloop()

