import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#=== Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, help="Filepath of the selected tensorboard csv file")
parser.add_argument("--xlabel", type=str, help="Name of the label on x-axis", default="x")
parser.add_argument("--ylabel", type=str, help="Name of the label on y-axis", default="y")
parser.add_argument("--title", type=str, help="Title of the plot",  default=" ")
parser.add_argument("--label", type=str, help="Label for the first plot", default="")
parser.add_argument("--filepath2", type=str, help="filepath of a second csv file to plot along the first", default="")
parser.add_argument("--label2", type=str, help="Label for the second plot")
parser.add_argument("--smoothing", type=float, help="Smoothing factor for plot, same as tensoboard", default=0.5)
parser.add_argument("--transparent", type=int, help="Plot transparent value of the plot", default=0)
args = parser.parse_args()
#============

def process_dt(filepath):
    headers = ['time', 'step', 'value']
    df = pd.read_csv(filepath, names=headers)

    #Remove headers
    df = df.iloc[1:]
    
    x0 = df['step'].astype(np.float32)
    y0 = df['value'].astype(np.float32)

    x0 = np.asarray(x0)
    y0 = np.asarray(y0)

    df = df.ewm(alpha=(1 - args.smoothing)).mean()
    #Get number of step (X axis), and associated value (Y axis)
    x = df['step'].astype(np.float32)
    y = df['value'].astype(np.float32)
    
    
    x = np.asarray(x)
    y = np.asarray(y)

    return x,y,x0,y0

x, y, x0, y0 = process_dt(args.filepath)


#Plot
if(args.transparent):
    plt.plot(x,y0,alpha=0.4)
plt.plot(x,y,label=args.label)
plt.title(args.title)
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

#In case of a 2nd plot
if args.filepath2 != "":
    x, y, x0, y0 = process_dt(args.filepath2)
    if(args.transparent):
        plt.plot(x,y0,alpha=0.4)
    plt.plot(x,y,label=args.label2)

plt.gcf().autofmt_xdate() #Nice
if args.label != "":
    plt.legend(loc='best')
plt.show()
