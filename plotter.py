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
	title = []
	xlabel = []
	ylabel = []
	
	annotations = [path for path in paths if "annotate" in open(path, "r").readline()]
	paths = [path for path in paths if "annotate" not in open(path, "r").readline()]
	
	for path in paths:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n")]
		headers = csv[0]
		graphs = parseGraphs(headers, csv[1:], name)
		if(graphs["title"] and graphs["title"] not in title):
			title.append(graphs["title"])
		if(graphs["xlabel"] and graphs["xlabel"] not in xlabel):
			xlabel.append(graphs["xlabel"])
		if(graphs["ylabel"] and graphs["ylabel"] not in ylabel):
			ylabel.append(graphs["ylabel"])
	
		for graph in graphs["graphs"]:
			x = graph["x"]
			y = graph["y"]
			label = graph["label"]
			if("vline" in graphs["options"]):
				ymin, ymax = ax.get_ylim()
				plots.append(ax.vlines(x, ymin, ymax, label = label))
				#for i, xval in enumerate(x):
					#ax.axvline(x = xval)
					#ax.annotate(y[i], (float(xval), float(ymin)), xytext = (float(xval), float(ymin) - (float(ymax) * 0.05)), horizontalalignment = 'right', verticalalignment = "top", rotation = 40, arrowprops={"arrowstyle": "->", "relpos": (1, 1)}, annotation_clip = False)
			#elif("annotate" in graphs["options"]):
			#	ymin, ymax = ax.get_ylim()
			#	for i, xval in enumerate(x):
			#		ax.annotate(y[i], (float(xval), float(ymin)), xytext = (float(xval), float(ymin) - (float(ymax) * 0.05)), horizontalalignment = 'right', verticalalignment = "top", rotation = 40, arrowprops={"arrowstyle": "->", "relpos": (1, 1)}, annotation_clip = False)
			elif("average" in graphs["options"]):
				averagedY, offsetStart, offsetEnd = moving_average(y, 5)
				#print(f"x len1: {len(x)}, len2: {len(x[offsetStart:-(offsetEnd)])}")
				#print(f"y len1: {len(y)}, len2: {len(y[offsetStart:-(offsetEnd)])}")
				plots.append(ax.plot(x, y, '.', label = f"{label}"))
				plots.append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}"))
			else:
				plots.append(ax.plot(x, y, '.-', label = label))
	
	ymin, ymax = ax.get_ylim()
	for path in annotations:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n")]
		headers = csv[0]
		graphs = parseGraphs(headers, csv[1:], name)
		if(graphs["title"] and graphs["title"] not in title):
			title.append(graphs["title"])
		if(graphs["xlabel"] and graphs["xlabel"] not in xlabel):
			xlabel.append(graphs["xlabel"])
		if(graphs["ylabel"] and graphs["ylabel"] not in ylabel):
			ylabel.append(graphs["ylabel"])
		
		for graph in graphs["graphs"]:
			x = graph["x"]
			y = graph["y"]
			label = graph["label"]
			colour = next(ax._get_lines.prop_cycler)['color']
			for i, xval in enumerate(x):
				ax.annotate(y[i], (float(xval), float(ymin)), xytext = (0, -50), horizontalalignment = 'right', verticalalignment = "top", rotation = 40, arrowprops={"arrowstyle": "->", "relpos": (1, 1), "alpha": 0.5, "color": colour}, annotation_clip = False, xycoords = "data", textcoords = "offset points")
	
	ax.legend()
	ax.set(xlabel = "|".join(xlabel), ylabel = " | ".join(ylabel), title = " | ".join(title))
	fig.savefig(getAvalableFilename(paths[0], "pdf"))
	plt.show()

Options = ["average", "generateX", "vline", "annotate"]

def parseGraphs(header, rows, filename = ""):
	graphs = {"graphs": [], "title": header[0], "xlabel": header[1], "ylabel": header[2], "options": []}
	
	num_columns = len(rows[0])
	i = 0
	x = []
	
	labels = [label for label in header[3:] if label not in Options]
	graphs["options"].extend([option for option in header[3:] if option in Options])
	
	vline = "vline" in graphs["options"]
	annotate = "annotate" in graphs["options"]
	generateX = ("generateX" in header[3:]) or num_columns == 1
	
	if(not len(labels)):
		labels.append(filename)
	
	if(not generateX):
		x = [float(column[0]) for column in rows]
		i = 1
	
	for label in labels:
		if(vline or annotate):
			x = [float(column[0]) for column in rows]
			y = [column[1] for column in rows]
			graphs["graphs"].append({"x": x, "y": y, "label": label})
			break
		else:
			y = [float(column[i]) for column in rows]
			if(generateX):
				x = range(len(y))
			graphs["graphs"].append({"x": x, "y": y, "label": label})
			i += 1
	return graphs

def main():
	if(len(sys.argv) > 2):
		GenerateGraphs(sys.argv[1:])
	else:
		GenerateGraphs([sys.argv[1]])

if __name__ == "__main__":
	main()
