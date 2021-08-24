import sys, ntpath
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

def moving_average(x, N, fill=True):
	averaged = np.concatenate([x for x in [ [None]*(N // 2 + N % 2)*fill, np.convolve(x, np.ones((N,))/N, mode='valid'), [None]*(N // 2)*fill, ] if len(x)])
	#print(f"a len1: {len(averaged)}, len2: {len(averaged[(N // 2 + N % 2):-(N // 2)])}")
	return averaged[(N // 2 + N % 2):-(N // 2)], (N // 2 + N % 2), (N // 2)

def getDPI():
	import tkinter
	root = tkinter.Tk()
	dpi = root.winfo_fpixels('1i')
	root.destroy()
	return dpi

def getAvalableFilename(path, ext):
	file = f"{path}.{ext}"
	if(ntpath.exists(file)):
		i = 1
		while(ntpath.exists(f"{path} ({i}).{ext}")):
			i += 1
		file = f"{path} ({i}).{ext}"
	return file

def GenerateGraphs(paths):
	plots = []
	dpi = getDPI()
	#fig, ax = plt.subplots(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
	fig, ax = plt.subplots(figsize=(3840/dpi, 2160/dpi), dpi=dpi)
	ax.yaxis.grid()
	headers = []
	for path in paths:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n")]
		headers = csv[0]
		if(len(csv[1:][0]) == 2):
			x, y = zip(*[list(map(float, (line[0], line[1]))) for line in csv[1:] if line[0] != "" and line[1] != ""])
		else:
			y = [float(line[0]) for line in csv[1:] if line[0] != ""]
			x = range(len(y))
		if(len(headers) == 3):
			plots.append(ax.plot(x, y, '.-', label = name)) #, linewidth = 1, markersize = 2
		elif(len(headers) > 3):
			if("average" in headers[3:]):
				averagedY, offsetStart, offsetEnd = moving_average(y, 5)
				#print(f"x len1: {len(x)}, len2: {len(x[offsetStart:-(offsetEnd)])}")
				#print(f"y len1: {len(y)}, len2: {len(y[offsetStart:-(offsetEnd)])}")
				plots.append(ax.plot(x, y, '.', label = f"{name}"))
				plots.append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{name}"))
	
	ax.legend()
	ax.set(xlabel=headers[1], ylabel=headers[2], title=headers[0])
	fig.savefig(getAvalableFilename(paths[0], "pdf"))
	plt.show()

def parseHeader(header, line1):
	pass

def main():
	if(len(sys.argv) > 2):
		GenerateGraphs(sys.argv[1:])
	else:
		GenerateGraphs([sys.argv[1]])

if __name__ == "__main__":
	main()
