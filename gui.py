import sys
import os
from tkinter import *

top = Tk()
top.title('GCP')


def calculator():
    os.system('python Calculator.py')


def mpc():
    os.system('python mpc.py')


def typepad():
    os.system('python Type-pad.py')


def vmc():
    os.system('python Virtual-Mouse-Control.py')


if __name__ == '__main__':

    B = Button(top, text="Calculator", command=calculator)
    B1 = Button(top, text="Media Control", command=mpc)
    B2 = Button(top, text="Typepad", command=typepad)
    B3 = Button(top, text="Virtual Mouse Control", command=vmc)
    B.grid(row=0, column=0, padx=20, pady=20, sticky='ew')
    B1.grid(row=0, column=1, padx=20, pady=20, sticky='ew')
    B2.grid(row=1, column=0, padx=20, pady=20, sticky='ew')
    B3.grid(row=1, column=1, padx=20, pady=20, sticky='ew')
    top.mainloop()
