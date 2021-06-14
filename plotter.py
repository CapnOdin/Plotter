import sys
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, N, fill=True):
	return np.concatenate([x for x in [ [None]*(N // 2 + N % 2)*fill, np.convolve(x, np.ones((N,))/N, mode='valid'), [None]*(N // 2)*fill, ] if len(x)]) 

def generateGraphs(paths):
	plots = []
	fig, ax = plt.subplots()
	ax.yaxis.grid()
	for path in paths:
		csv = open(path, "r").readlines()
		sf, cr, bw = csv[0].replace("\n", "").split(",")
		x, y = zip(*[list(map(float, line.split(","))) for line in csv[1:]])
		plots.append(ax.plot(x, y, '.-', label=bw))
	
	#ax.legend(handles=plots)
	ax.legend()
	ax.set(xlabel='msg lenght', ylabel='time (ms)',
	       title=f'Tx delay over msg size - {sf} - {cr}')
	fig.savefig(f"LoRa tx time - {sf} - {cr}.png")
	plt.show()

def generateGraph(path):
	csv = open(path, "r").readlines()
	sf, cr, bw = csv[0].replace("\n", "").split(",")
	x, y = zip(*[list(map(int, line.split(","))) for line in csv[1:]])
	
	fig, ax = plt.subplots()
	ax.plot(x, y, '.-')
	ax.set(xlabel='msg lenght', ylabel='time (ms)',
	       title=f'Tx delay over msg size - {sf} - {cr} - {bw}')
	fig.savefig(f"LoRa tx time - {sf} - {bw}.png")
	plt.show()
	
def generateGraph2(path):
	plots = []
	fig, ax = plt.subplots()
	ax.yaxis.grid()
	csv = [line.split(",") for line in open(path, "r").read().split("\n")]
	headers = csv[0]
	for i, head in enumerate(headers[1:]):
		x, y = zip(*[list(map(float, (line[0], line[i+1]))) for line in csv[1:]])
		plots.append(ax.plot(x, y, '.-', label=head))
	
	ax.legend()
	ax.set(xlabel='msg lenght', ylabel='time (ms)',
	       title=f'Tx delay over msg size')
	fig.savefig(f"LoRa tx time.png")
	plt.show()
	
def generateGraph3(path):
	plots = []
	fig, ax = plt.subplots()
	ax.yaxis.grid()
	csv = [line.split(",") for line in open(path, "r").read().split("\n")]
	headers = csv[0]
	for i, head in enumerate(headers[1:]):
		x, y = zip(*[list(map(float, (line[0], line[i+1]))) for line in csv[1:]])
		#npy = np.array(y)
		N = len(y)
		npy = moving_average(y, 5)
		#npy = np.convolve(y, np.ones((N,))/N, mode='valid')
		print(npy)
		plots.append(ax.plot(x, npy[0:100], '.-', label=head))
	
	ax.legend()
	ax.set(xlabel='msg lenght', ylabel='time (ms)',
	       title=f'Tx delay over msg size')
	fig.savefig(f"LoRa tx time.png")
	plt.show()

def main():
	if(len(sys.argv) > 2):
		generateGraphs(sys.argv[1:])
	else:
		generateGraph2(sys.argv[1])

if __name__ == "__main__":
	main()
