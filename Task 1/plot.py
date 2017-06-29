import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import argparse

parser = argparse.ArgumentParser('Generate plots for the input files.')
parser.add_argument('--files', nargs='+', help='List of files to plot', required=True)
parser.add_argument('--options', type=int, help='Plot train and val in the same plot:1. Different plots: 0', required=True)
parser.add_argument('--name', help='Name for the output files', required=True)


args = parser.parse_args()

# values[k]: 0-> training acc  1-> training loss   2-> val acc     3-> val loss
values = []

for file in args.files:
    with open(file, 'r') as f:
        data = []
        for i in range(4):
            line = f.readline()[1:-2].split(", ")
            data.append(np.array([float(i) for i in line]))
        values.append(data)


colors = ['r', 'b', 'g', 'c', 'y', 'k']

#Print train accuracy
for i, data in enumerate(values):
    plt.plot(data[0], label=(args.files[i].split(".")[0] + '_train'))

plt.title('Accuracy')
plt.xlabel('Steps')
plt.ylabel('Top-1 Accuracy')
plt.grid('on')

plt.ylim([0.4,1.1])

if args.options is 0:
    plt.legend()
    plt.savefig(args.name + 'accuracy.png')
    plt.close()
    plt.title('Accuracy val')
    plt.xlabel('Steps')
    plt.ylabel('Top-1 Accuracy val')
    plt.grid('on')

    for i, data in enumerate(values):
        plt.plot(data[2], label=(args.files[i].split(".")[0] + '_val'))

    plt.legend()
    plt.savefig(args.name + 'accuracy_val.png', dpi=300)

else:
    for i, data in enumerate(values):
        plt.plot(data[2], label=(args.files[i].split(".")[0] + '_val'))

    plt.legend()
    plt.savefig(args.name + 'accuracy.png', dpi=300)

plt.close()

# Print train loss
for i, data in enumerate(values):
    plt.plot(data[1], label=(args.files[i].split(".")[0] + '_train'))

plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid('on')


if args.options is 0:
    plt.legend()
    plt.savefig(args.name + 'loss.png', dpi=300)
    plt.close()
    plt.title('Loss val')
    plt.xlabel('Steps')
    plt.ylabel('Loss val')
    plt.grid('on')
    for i, data in enumerate(values):
        plt.plot(data[3], label=(args.files[i].split(".")[0] + '_val'))

    plt.legend()
    plt.savefig(args.name + 'loss_val.png', dpi=300)

else:
    for i, data in enumerate(values):
        plt.plot(data[3], label=(args.files[i].split(".")[0] + '_val'))
    plt.legend()
    plt.savefig(args.name + 'loss.png', dpi=300)