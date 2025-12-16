from enum import Enum
from functools import total_ordering
import sys, ntpath, argparse, traceback, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
#from sklearn import preprocessing
import datetime as dt
import matplotlib.dates as md
import matplotlib.transforms as mtrans
from matplotlib.axes import Axes

from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from typing import List, Any

from pprint import pformat
from textwrap import indent
import json
from datetime import timedelta

#np.set_printoptions(threshold=sys.maxsize)

def clamp(num, min_value, max_value):
	num = max(min(num, max_value), min_value)
	return num

cumulativeLineOffset = 0

@total_ordering
class Orientation(Enum):
	NONE		= -1
	VERTICAL	=  0
	HORIZONTAL	=  1

	def __lt__(self, other):
		if(self.__class__ is other.__class__):
			return self.value < other.value
		return NotImplemented
	
	def byValue(value):
		return list(Orientation)[clamp(value + abs(next(Orientation.__iter__()).value), 0, len(list(Orientation)))]

def moving_average(x, N, fill=True):
	averaged = np.concatenate([x for x in [[None]*(N // 2 + N % 2)*fill, np.convolve(x, np.ones((N,))/N, mode='valid'), [None]*(N // 2)*fill, ] if len(x)])
	#print(f"a len1: {len(averaged)}, len2: {len(averaged[(N // 2 + N % 2):-(N // 2)])}")
	return averaged[(N // 2 + N % 2):-(N // 2)], (N // 2 + N % 2), (N // 2)

def NormalizeData(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def getTitle(titles = False, filename = "", graphTitles = [], index = 0, indent: str = ""):
	print(f"\n\n{indent}{titles = }\n{indent}{filename = }\n{indent}{graphTitles = }\n{indent}{index = }")
	if(titles):
		if(type(titles) == str):
			return titles
		elif(len(titles) >= index):
			return titles[index]
	elif(len(graphTitles)):
		return " | ".join(graphTitles)
	else:
		return filename

def GeneratePlotlyGraphs(paths, outdir = "", title = "", interactive: bool = False):
	if(title):
		fileName = title
	else:
		fileName = os.path.split(paths[0])[1]
	
	if(not outdir):
		outdir = os.path.split(paths[0])[0]
		
	print(f"{outdir}\n{fileName}")
		
	plots = []
	dpi = getDPI()
	#fig, ax = plt.subplots(figsize = (1920 / dpi, 1080 / dpi), dpi = dpi)
	fig, ax = plt.subplots(figsize = (3840 / dpi, 2160 / dpi), dpi = dpi)
	fig = go.Figure()
	ax.yaxis.grid()
	titles = []
	xlabel = []
	ylabel = []

	xlim = {"min": [], "max": []}
	xExplisitLim = {"min": [], "max": []}
	
	annotations = [path for path in paths if "annotate" in open(path, "r").readline()]
	paths = [path for path in paths if "annotate" not in open(path, "r").readline()]
	
	for path in paths:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
		headers = csv[0]
		if( not len(csv[1:])):
			print(f"Skipped Empty File: {path}")
			continue
		graphs = parseGraphs(headers, csv[1:], name)
		if(graphs["title"] and graphs["title"] not in titles):
			titles.append(graphs["title"])
		if(graphs["xlabel"] and graphs["xlabel"] not in xlabel):
			xlabel.append(graphs["xlabel"])
		if(graphs["ylabel"] and graphs["ylabel"] not in ylabel):
			ylabel.append(graphs["ylabel"])
	
		for graph in graphs["graphs"]:
			x = graph["x"]
			y = graph["y"]
			label = graph["label"]
			if("vline" in graphs["options"]):
				pass
			elif("vColourArea" in graphs["options"]):
				pass
			elif("xAxisSized" in graphs["options"]):
				xExplisitLim["min"].append(min(x))
				xExplisitLim["max"].append(max(x))
			elif("average" in graphs["options"]):
				averagedY, offsetStart, offsetEnd = moving_average(y, 12)
				if("dots" in graphs["options"]):
					fig.add_trace(go.Scatter(
						x = x,
						y = y,
						mode = 'markers',
						name = f"{label}"
					))
					print(fig.data[-1])
					fig.add_trace(go.Line(
						x = x[offsetStart - 1:-offsetEnd],
						y = averagedY,
						name = f"averaged-{label}",
						line = dict(color = fig.data[-1].line.color)
					))
				else:
					fig.add_trace(go.Line(
						x = x[offsetStart - 1:-offsetEnd],
						y = averagedY,
						name = f"averaged-{label}"
					))
			elif("bar" in graphs["options"]):
				fig.add_trace(go.Bar(
					x = x,
					y = y,
					name = label
				))
			elif("dots" in graphs["options"]):
				fig.add_trace(go.Scatter(
					x = x,
					y = y,
					mode = 'markers',
					name = label
				))
			elif("line" in graphs["options"]):
				fig.add_trace(go.Line(
					x = x,
					y = y,
					name = label
				))
			else:
				fig.add_trace(go.Line(
					x = x,
					y = y,
					mode = 'lines+markers',
					name = label,
					
				))
			xlim["min"].append(min(x))
			xlim["max"].append(max(x))
	fig.write_html(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "html"))
	if(interactive):
		fig.show()

class Colours:
	def __init__(self, colours: List[Any]) -> None:
		self.colours = colours
		self.index = 0
	
	def next(self):
		self.index = (self.index + 1) % len(self.colours)
		return self.colours[self.index]

def getNextColor(colors: List[Any], index: int):
	return colors[index + 1]

def GenerateMultiPlotlyGraphs(paths, orientation, outdir = "", title = "", interactive: bool = False):
	if(title):
		fileName = title if type(title) == str else title[0]
	else:
		fileName = os.path.split(paths[0][0])[1]
	
	if(not outdir):
		outdir = os.path.split(paths[0][0])[0]
	
	plots = []
	dpi = getDPI()
	if(orientation == Orientation.VERTICAL):
		fig = go.Figure()
		fig = make_subplots(rows=1, cols=2)
		fig, axs = plt.subplots(len(paths), figsize = (1920 / dpi, 1080 / dpi), dpi = dpi)
		#fig, axs = plt.subplots(len(paths), figsize = (3840 / dpi, 2160 / dpi), dpi = dpi)
		fig = make_subplots(rows = len(paths), cols = 1)
	else:
		#fig, axs = plt.subplots(1, len(paths), figsize = (1920 / dpi, 1080 / dpi), dpi = dpi)
		fig, axs = plt.subplots(1, len(paths), figsize = (3840 / dpi, 2160 / dpi), dpi = dpi)
		fig = make_subplots(rows = 1, cols = len(paths))

	print(type(axs))
	print(f"ndarray: {type(axs) is np.ndarray}")
	
	if(type(axs) is np.ndarray):
		axs = axs.flat
	else:
		axs = [axs]

	for j, ax in enumerate(axs):
		ax.yaxis.grid()
		titles = []
		xlabel = []
		ylabel = []

		xlim = {"min": [], "max": []}
		xExplisitLim = {"min": [], "max": []}
		
		annotations = [path for path in paths[j] if "annotate" in open(path, "r").readline()]
		localpaths = [path for path in paths[j] if "annotate" not in open(path, "r").readline()]
		
		print(f"SubPlot {j}:")

		row = j + 1 if orientation == Orientation.VERTICAL else 1
		col = j + 1 if orientation == Orientation.HORIZONTAL else 1

		colors = Colours(DEFAULT_PLOTLY_COLORS)
		
		for path in localpaths:
			name = ntpath.basename(path)
			csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
			headers = csv[0]
			if(not len(csv[1:])):
				print(f"Skipped Empty File: {path}")
				continue
			graphs = parseGraphs(headers, csv[1:], name)
			if(graphs["title"] and graphs["title"] not in titles):
				titles.append(graphs["title"])
			if(graphs["xlabel"] and graphs["xlabel"] not in xlabel):
				xlabel.append(graphs["xlabel"])
			if(graphs["ylabel"] and graphs["ylabel"] not in ylabel):
				ylabel.append(graphs["ylabel"])

			if("xAxisTicks" in graphs["options"]):
				arg = {f"xaxis{row if Orientation.VERTICAL else col}": 
					dict(
						tickmode = 'array',
						tickvals = graphs["graphs"][0]["ticks"],
						ticktext = graphs["graphs"][0]["labels"]
					)
				}
				fig.update_layout(**arg)
				continue
			elif("yAxisTicks" in graphs["options"]):
				arg = {f"yaxis{row if Orientation.VERTICAL else col}": 
					dict(
						tickmode = 'array',
						tickvals = graphs["graphs"][0]["ticks"],
						ticktext = graphs["graphs"][0]["labels"]
					)
				}
				fig.update_layout(**arg)
				continue
		
			for graph in graphs["graphs"]:
				x = graph["x"]
				y = graph["y"]
				label = graph["label"]
				if("vline" in graphs["options"]):
					color = colors.next()
					print(f"\t\tvline: {color=:}")
					for pos in x:
						fig.add_vline(x = pos, row = row, col = col, line_color = color, line_width = 3, exclude_empty_subplots = False) # layer = "between"
				elif("vColourArea" in graphs["options"]):
					color = colors.next()
					print(f"\t\tvColourArea: {color = }")
					for xval, yval in zip(x, y):
						fig.add_vrect(x0 = xval, x1 = yval, row = row, col = col, opacity = 0.25, line_width = 0, fillcolor = color, layer = "below")
				elif("xAxisSized" in graphs["options"]):
					xExplisitLim["min"].append(min(x))
					xExplisitLim["max"].append(max(x))
				elif("average" in graphs["options"]):
					color = colors.next()
					averagedY, offsetStart, offsetEnd = moving_average(y, 12)
					if("dots" in graphs["options"]):
						fig.add_trace(go.Scatter(
							x = x,
							y = y,
							mode = 'markers',
							name = f"{label}",
							color = color
						), row = row, col = col)
						print(fig.data[-1])
						fig.add_trace(go.Line(
							x = x[offsetStart - 1:-offsetEnd],
							y = averagedY,
							name = f"averaged-{label}",
							line = dict(color = fig.data[-1].line.color)
						), row = row, col = col)
					else:
						fig.add_trace(go.Line(
							x = x[offsetStart - 1:-offsetEnd],
							y = averagedY,
							name = f"averaged-{label}",
							color = color
						), row = row, col = col)
				elif("bar" in graphs["options"]):
					color = colors.next()
					fig.add_trace(go.Bar(
						x = x,
						y = y,
						name = label,
						marker_color = color,
						width = 1,
						opacity = 0.50
					), row = row, col = col)
				elif("dots" in graphs["options"]):
					color = colors.next()
					fig.add_trace(go.Scatter(
						x = x,
						y = y,
						mode = 'markers',
						name = label,
						fillcolor = color
					), row = row, col = col)
				elif("line" in graphs["options"]):
					color = colors.next()
					fig.add_trace(go.Line(
						x = x,
						y = y,
						name = label,
						line = dict(color = color)
					), row = row, col = col)
				else:
					color = colors.next()
					fig.add_trace(go.Line(
						x = x,
						y = y,
						mode = 'lines+markers',
						name = label,
						line = dict(color = color)
					), row = row, col = col)
				xlim["min"].append(min(x))
				xlim["max"].append(max(x))
	
	if(xExplisitLim["min"] or xExplisitLim["max"]):
		xaxis_range = [min(xExplisitLim["min"]) if xExplisitLim["min"] else None, max(xExplisitLim["max"]) if xExplisitLim["max"] else None]
		#fig.update_xaxes(range = xaxis_range, type = 'category')
		fig.update_xaxes(range = xaxis_range)

	#millis = (((max(xlim["max"]) - min(xlim["min"])) / 100)) / timedelta(milliseconds = 1)
	#fig.update_traces(selector = dict(type = "bar"), width = millis)

	fig.update_xaxes(showgrid = False)

	#fig.update_layout(barmode = 'group', bargap = 0.9, autosize = True)
	#fig.update_layout(barmode = 'group', bargap = 0.9)
	#fig.update_layout(bargap = 0.9)
		
	fig.write_html(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "html"))
	if(interactive):
		fig.show()

def mscatter(x, y, ax = None, markers = None, **kw):
	if not ax:
		ax = plt.gca()
	sc = ax.scatter(x, y, **kw)
	if((markers is not None) and (len(markers) == len(x))):
		paths = []
		for marker in markers:
			if isinstance(marker, MarkerStyle):
				marker_obj = marker
			else:
				marker_obj = MarkerStyle(marker)
			path = marker_obj.get_path().transformed(marker_obj.get_transform())
			paths.append(path)
		sc.set_paths(paths)
	return sc

def GenerateGraphs(paths, outdir = "", title = ""):
	if(title):
		fileName = title
	else:
		fileName = os.path.split(paths[0])[1]
	
	if(not outdir):
		outdir = os.path.split(paths[0])[0]
		
	print(f"{outdir}\n{fileName}")
	
	plots = []
	dpi = getDPI()
	#fig, ax = plt.subplots(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
	fig, ax = plt.subplots(figsize=(3840/dpi, 2160/dpi), dpi=dpi)
	ax.yaxis.grid()
	titles = []
	xlabel = []
	ylabel = []

	xlim = {"min": [], "max": []}
	xExplisitLim = {"min": [], "max": []}
	
	annotations = [path for path in paths if "annotate" in open(path, "r").readline()]
	paths = [path for path in paths if "annotate" not in open(path, "r").readline()]
	
	for path in paths:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
		headers = csv[0]
		if( not len(csv[1:])):
			print(f"Skipped Empty File: {path}")
			continue
		graphs = parseGraphs(headers, csv[1:], name)
		if(graphs["title"] and graphs["title"] not in titles):
			titles.append(graphs["title"])
		if(graphs["xlabel"].strip() and graphs["xlabel"] not in xlabel):
			xlabel.append(graphs["xlabel"])
		if(graphs["ylabel"].strip() and graphs["ylabel"] not in ylabel):
			ylabel.append(graphs["ylabel"])
		
		if("xAxisTicks" in graphs["options"]):
			ax.set_xticks(graphs["graphs"][0]["ticks"], labels = graphs["graphs"][0]["labels"])
			continue
		elif("yAxisTicks" in graphs["options"]):
			ax.set_yticks(graphs["graphs"][0]["ticks"], labels = graphs["graphs"][0]["labels"])
			continue

		for graph in graphs["graphs"]:
			x = graph["x"]
			y = graph["y"]
			label = graph["label"]
			if("vline" in graphs["options"]):
				color = next(ax._get_lines.prop_cycler)["color"]
				print(f"vline: {color=:}")
				for pos in x:
					plots.append(ax.axvline(pos, color = color, label = label, zorder = -100))
			elif("vColourArea" in graphs["options"]):
				color = next(ax._get_lines.prop_cycler)["color"]
				print(f"vColourArea: {color=:}")
				for xval, yval in zip(x, y):
					plots.append(ax.axvspan(xval, yval, facecolor = color, alpha = 0.25, zorder = -100))
			elif("xAxisSized" in graphs["options"]):
				xExplisitLim["min"].append(min(x))
				xExplisitLim["max"].append(max(x))
			elif("average" in graphs["options"]):
				averagedY, offsetStart, offsetEnd = moving_average(y, 12)
				if("dots" in graphs["options"]):
					#if(graphs["graphs"][0] != graph):
					#	colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
					#	ax2 = ax.twiny()
					#	plots.append(ax2.plot(x, y, '.', label = f"{label}", color = colours[1]))
					#	plots.append(ax2.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}", color = plots[-1][0].get_color()))
					#else:
					plots.append(ax.plot(x, y, '.', label = f"{label}"))
					plots.append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}", color = plots[-1][0].get_color()))
				else:
					plots.append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}"))
			elif("heatmap" in graphs["options"]):
				im = ax.imshow(graph["m"])
				fig.colorbar(im)
				plots.append(plt)
			elif("histogram" in graphs["options"]):
				plots.append(ax.hist(y, bins = 50, facecolor = 'blue', alpha = 0.5, edgecolor = "white"))
			elif("bar" in graphs["options"]):
				color = next(ax._get_lines.prop_cycler)["color"]
				print(f"bar: {color=:}")
				plots.append(ax.bar(x, y, width = 0.001, alpha = 0.5, color = color, edgecolor = "white", label = label))
			elif("dots" in graphs["options"]):
				plots.append(ax.plot(x, y, '.', label = label))
			elif("crosses" in graphs["options"]):
				plots.append(ax.plot(x, y, 'x', label = label))
			elif("line" in graphs["options"]):
				plots.append(ax.plot(x, y, '-', label = label))
			else:
				plots.append(ax.plot(x, y, '.-', label = label))
			xlim["min"].append(min(x))
			xlim["max"].append(max(x))

	ymin, ymax = ax.get_ylim()
	for path in annotations:
		name = ntpath.basename(path)
		csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
		headers = csv[0]
		if( not len(csv[1:])):
			print(f"Skipped Empty File: {path}")
			continue
		graphs = parseGraphs(headers, csv[1:], name)
		if(graphs["title"] and graphs["title"] not in titles):
			titles.append(graphs["title"])
		if(graphs["xlabel"].strip() and graphs["xlabel"] not in xlabel):
			xlabel.append(graphs["xlabel"])
		if(graphs["ylabel"].strip() and graphs["ylabel"] not in ylabel):
			ylabel.append(graphs["ylabel"])
		
		fig.subplots_adjust(bottom = 0.2)
		
		for graph in graphs["graphs"]:
			x = graph["x"]
			y = graph["y"]
			label = graph["label"]
			colour = next(ax._get_lines.prop_cycler)['color']
			xlim["min"].append(min(x))
			xlim["max"].append(max(x))
			for i, xval in enumerate(x):
				ax.annotate(y[i], (xval, float(ymin)), xytext = (0, -50), horizontalalignment = 'right', verticalalignment = "top", rotation = 40, arrowprops={"arrowstyle": "->", "relpos": (1, 1), "alpha": 0.5, "color": colour}, annotation_clip = False, xycoords = "data", textcoords = "offset points")
	
	if(xExplisitLim["min"] or xExplisitLim["max"]):
		ax.set_xlim(min(xExplisitLim["min"]) if xExplisitLim["min"] else None, max(xExplisitLim["max"]) if xExplisitLim["max"] else None)
	elif(len(annotations)):
		ax.set_xlim(min(xlim["min"]), max(xlim["max"]))
	
	try:
		ax.ticklabel_format(useOffset=False, style='plain')
	except:
		pass

	ax.legend()
	ax.set(xlabel = "|".join(xlabel), ylabel = " | ".join(ylabel), title = getTitle(title, fileName, titles))
	if(title):
		fig.canvas.manager.set_window_title(" ".join(title) if type(title) == list else title)
	fig.savefig(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "pdf"))
	#fig.savefig(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "png"))
	plt.show()

import matplotlib.dates as mdates
import matplotlib.units as munits

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[dt.date] = converter
munits.registry[dt.datetime] = converter

def GenerateMultiGraphs(paths, orientation, outdir = "", title = "", interactive: bool = False):
	global cumulativeLineOffset
	if(title):
		fileName = title if type(title) == str else title[0]
	else:
		fileName = os.path.split(paths[0][0])[1]
	
	if(not outdir):
		outdir = os.path.split(paths[0][0])[0]
	
	plots = []
	dpi = getDPI()
	if(orientation == Orientation.VERTICAL):
		fig, axs = plt.subplots(len(paths), figsize = (1920 / dpi, 1080 / dpi), dpi = dpi)
		#fig, axs = plt.subplots(len(paths), figsize = (3840 / dpi, 2160 / dpi), dpi = dpi)
	else:
		#fig, axs = plt.subplots(1, len(paths), figsize = (1920 / dpi, 1080 / dpi), dpi = dpi)
		fig, axs = plt.subplots(1, len(paths), figsize = (3840 / dpi, 2160 / dpi), dpi = dpi)

	print(type(axs))
	print(f"ndarray: {type(axs) is np.ndarray}")
	
	if(type(axs) is np.ndarray):
		axs = axs.flat
	else:
		axs = [axs]

	for j, ax in enumerate(axs):
		plots.append([])
		ax: Axes
		ax.yaxis.grid(zorder = -50)
		titles = []
		xlabel = []
		ylabel = []

		xlim = {"min": [], "max": []}
		xExplisitLim = {"min": [], "max": []}
		yExplisitLim = {"min": [], "max": []}
		
		annotations = [path for path in paths[j] if "annotate" in open(path, "r").readline()]
		localpaths = [path for path in paths[j] if "annotate" not in open(path, "r").readline()]
		
		cumulativeLineOffset = 0

		print(f"SubPlot {j}:")
		
		for path in localpaths:
			try:
				name = ntpath.basename(path)
				csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
				headers = csv[0]
				if(not len(csv[1:])):
					print(f"Skipped Empty File: {path}")
					continue
				graphs = parseGraphs(headers, csv[1:], name)
				if(graphs["title"] and graphs["title"] not in titles):
					titles.append(graphs["title"])
				if(graphs["xlabel"].strip() and graphs["xlabel"] not in xlabel):
					xlabel.append(graphs["xlabel"])
				if(graphs["ylabel"].strip() and graphs["ylabel"] not in ylabel):
					ylabel.append(graphs["ylabel"])
				
				print(f'\t{name = }, {graphs["title"] = }, {graphs["xlabel"] = }, {graphs["ylabel"] = }, {graphs["options"] = }')
				#print(indent(pformat(graphs), "\t\t\t"))

				if("xAxisTicks" in graphs["options"]):
					ax.set_xticks(graphs["graphs"][0]["ticks"], labels = graphs["graphs"][0]["labels"])
					continue
				elif("yAxisTicks" in graphs["options"]):
					ax.set_yticks(graphs["graphs"][0]["ticks"], labels = graphs["graphs"][0]["labels"])
					continue

				for graph in graphs["graphs"]:
					x = graph["x"]
					y = graph["y"]
					label = graph["label"]
					if("vline" in graphs["options"]):
						# TODO: try using ax._get_lines.get_next_color() see if it works in all cases
						color = next(ax._get_lines.prop_cycler)["color"]
						print(f"\t\tvline: {color = }")
						for pos in x:
							plots[-1].append(ax.axvline(pos, color = color, label = label, zorder = -100))
					elif("vColourArea" in graphs["options"]):
						color = next(ax._get_lines.prop_cycler)["color"]
						print(f"\t\tvColourArea: {color = }")
						for xval, yval in zip(x, y):
							plots[-1].append(ax.axvspan(xval, yval, color = color, alpha = 0.25, zorder = -100))
					elif("xAxisSized" in graphs["options"]):
						xExplisitLim["min"].append(min(x))
						xExplisitLim["max"].append(max(x))
					elif("yAxisSized" in graphs["options"]):
						yExplisitLim["min"].append(min(x))
						yExplisitLim["max"].append(max(x))
					elif("average" in graphs["options"]):
						averagedY, offsetStart, offsetEnd = moving_average(y, 12)
						if("dots" in graphs["options"]):
							#if(graphs["graphs"][0] != graph):
							#	colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
							#	ax2 = ax.twiny()
							#	plots[-1].append(ax2.plot(x, y, '.', label = f"{label}", color = colours[1]))
							#	plots[-1].append(ax2.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}", color = plots[-1][-1][0].get_color()))
							#else:
							plots[-1].append(ax.plot(x, y, '.', label = f"{label}"))
							plots[-1].append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}", color = plots[-1][-1][0].get_color()))
						else:
							plots[-1].append(ax.plot(x[offsetStart - 1:-offsetEnd], averagedY, '-', label = f"averaged-{label}"))
					elif("marker" in graphs["options"]):
						color = next(ax._get_lines.prop_cycler)["color"]
						plots[-1].append(ax.plot(x, y, '-', label = label, color = color))
						plots[-1].append(mscatter(x, y, ax = ax, markers = graph["marker"], color = color, clip_on = False))
					elif("colour" in graphs["options"]):
						color = next(ax._get_lines.prop_cycler)["color"]
						plots[-1].append(ax.plot(x, y, '-', label = label, color = color))
						plots[-1].append(ax.scatter(x, y, c = graph["colour"], zorder = plots[-1][-1][0].get_zorder() + 1, clip_on = False))
					elif("heatmap" in graphs["options"]):
						im = ax.imshow(graph["m"])
						fig.colorbar(im)
						plots[-1].append(plt)
					elif("histogram" in graphs["options"]):
						#plots[-1].append(ax.hist(y, bins = 5939, facecolor = 'blue', alpha = 0.5, edgecolor = "white"))
						plots[-1].append(ax.hist(y, bins = 50, facecolor = 'blue', alpha = 0.5, edgecolor = "white", weights = y))
					elif("bar" in graphs["options"]):
						color = next(ax._get_lines.prop_cycler)["color"]
						print(f"\t\tbar: {color = }")
						plots[-1].append(ax.bar(x, y, width = 0.001, alpha = 0.5, color = color, edgecolor = "white", label = label))
					elif("dots" in graphs["options"]):
						plots[-1].append(ax.plot(x, y, '.', label = label))
					elif("crosses" in graphs["options"]):
						plots[-1].append(ax.plot(x, y, 'x', label = label))
					elif("line" in graphs["options"]):
						plots[-1].append(ax.plot(x, y, '-', label = label))
					else:
						plots[-1].append(ax.plot(x, y, '.-', label = label))
					xlim["min"].append(min(x))
					xlim["max"].append(max(x))
			except Exception as e:
				print(f"Failed Plotting {path = }")
				raise e
		
		if(cumulativeLineOffset):
			plot: Line2D
			offset = cumulativeLineOffset / 2
			for plot in plots[-1]:
				if(isinstance(plot, List)):
					for plot2 in plot:
						if(isinstance(plot2, Line2D)):
							plot2.set_ydata([y - offset for y in plot2.get_ydata(orig = True)])
			
				if(isinstance(plot, PathCollection)):
					plot: PathCollection
					offsets = plot.get_offsets().data
					plot.set_offsets(offsets + (0, -offset))
					plot.set_zorder(plot.get_zorder() + 10)
			
			fig.canvas.draw()
			fig.canvas.flush_events()

		ymin, ymax = ax.get_ylim()
		for path in annotations:
			name = ntpath.basename(path)
			csv = [line.split(",") for line in open(path, "r").read().split("\n") if line != ""]
			headers = csv[0]
			if(not len(csv[1:])):
				print(f"Skipped Empty File: {path}")
				continue
			graphs = parseGraphs(headers, csv[1:], name)
			if(graphs["title"] and graphs["title"] not in titles):
				titles.append(graphs["title"])
			if(graphs["xlabel"].strip() and graphs["xlabel"] not in xlabel):
				xlabel.append(graphs["xlabel"])
			if(graphs["ylabel"].strip() and graphs["ylabel"] not in ylabel):
				ylabel.append(graphs["ylabel"])
			
			fig.subplots_adjust(bottom = 0.2)

			trans = mtrans.blended_transform_factory(ax.transAxes, fig.transFigure)
			clippath = plt.Rectangle((0,0), 1, 1, transform=trans, clip_on = False)
			
			for graph in graphs["graphs"]:
				x = graph["x"]
				y = graph["y"]
				label = graph["label"]
				colour = next(ax._get_lines.prop_cycler)['color']
				xlim["min"].append(min(x))
				xlim["max"].append(max(x))
				for i, xval in enumerate(x):
					ano = ax.annotate(y[i], (xval, float(ymin)), xytext = (0, -50), horizontalalignment = 'right', verticalalignment = "top", rotation = 40, arrowprops = {"arrowstyle": "->", "relpos": (1, 1), "alpha": 0.5, "color": colour}, annotation_clip = True, xycoords = "data", textcoords = "offset points")
					ano.set_clip_path(clippath)
		
		if(yExplisitLim["min"] or yExplisitLim["max"]):
			ax.set_ylim(min(yExplisitLim["min"]) - (cumulativeLineOffset / 2) if yExplisitLim["min"] else None, max(yExplisitLim["max"]) + (cumulativeLineOffset / 2) if yExplisitLim["max"] else None)
		if(xExplisitLim["min"] or xExplisitLim["max"]):
			ax.set_xlim(min(xExplisitLim["min"]) if xExplisitLim["min"] else None, max(xExplisitLim["max"]) if xExplisitLim["max"] else None)
		elif(len(annotations)):
			ax.set_xlim(min(xlim["min"]), max(xlim["max"]))
		
		try:
			ax.ticklabel_format(useOffset = False, style = 'plain')
		except:
			pass

		ax.legend()
		ax.set(xlabel = " | ".join(xlabel), ylabel = " | ".join(ylabel), title = getTitle(title, fileName, titles, j, indent = "\t\t"))
		ax.set_axisbelow(True)
	if(title):
		fig.canvas.manager.set_window_title(" ".join(title) if type(title) == list else title)
	fig.tight_layout()
	#fig.savefig(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "pdf"))
	fig.savefig(getAvalableFilename(f"{outdir}{os.sep}{fileName}", "png"))
	if(interactive):
		plt.show()

Options = ["average", "generateX", "vline", "annotate", "dots", "crosses", "line", "heatmap", "normalize", "histogram", "bar", "unixtime", "vColourArea", "xAxisSized", "yAxisSized", "xAxisTicks", "yAxisTicks", "nolabel", "grouped", "marker", "colour", "offset"]

def parseGraphs(header, rows, filename = ""):
	global cumulativeLineOffset
	try:
		graphs = {"graphs": [], "title": header[0], "xlabel": header[1], "ylabel": header[2], "options": []}
		
		num_columns = len(rows[0])
		num_rows = len(rows)
		i = 0
		x = []
		
		labels = [label for label in header[3:] if label not in Options]
		graphs["options"].extend([option for option in header[3:] if option in Options])
		
		vline = "vline" in graphs["options"]
		annotate = "annotate" in graphs["options"]
		xAxisSized = "xAxisSized" in header[3:]
		yAxisSized = "yAxisSized" in header[3:]
		xAxisTicks = "xAxisTicks" in header[3:]
		yAxisTicks = "yAxisTicks" in header[3:]
		unixTimeStamp = "unixtime" in header[3:]
		generateX = ("generateX" in header[3:]) or (num_columns == 1 and not xAxisSized and not yAxisSized and not unixTimeStamp and not xAxisTicks and not yAxisTicks)
		heatmap = "heatmap" in header[3:]
		colourArea = "vColourArea" in header[3:]
		nolabel = "nolabel" in header[3:]
		marker = "marker" in header[3:]
		colour = "colour" in header[3:]
		offset = "offset" in header[3:]
		
		#rows = [[x if x != "" and x != None else np.nan for x in row] for row in rows]
		
		try:
			non_str_rows = [list(map(float, x)) for x in rows]
			
			if("normalize" in header[3:]):
				non_str_rows = [list(x) for x in NormalizeData(non_str_rows)]
		except:
			print("Could not convert to non str")
			
		if(not len(labels)):
			labels.append(filename)
		
		if(not generateX):
			if(unixTimeStamp):
				x = [dt.datetime.fromtimestamp(float(column[0]) / 1000.0, tz = dt.timezone.utc) if column[0] else np.nan for column in rows]
				if(colourArea):
					y = [dt.datetime.fromtimestamp(float(column[1]) / 1000.0, tz = dt.timezone.utc) for column in rows]
			else:
				x = [float(column[0]) for column in rows]
			i = 1
		
		if(marker):
			markers = [column[-1] if len(column[-1]) > 0 else np.nan for column in rows if len(column)]
		if(colour):
			colours = [column[-1] if len(column[-1]) > 0 else np.nan for column in rows if len(column)]
		
		for label in labels:
			if(annotate):
				y = [column[1] for column in rows]
				graphs["graphs"].append({"x": x, "y": y, "label": label})
				break
			elif(vline):
				graphs["graphs"].append({"x": x, "y": None, "label": label})
				break
			elif(heatmap):
				m = np.array(np.array(list(map(float, rows[0]))))
				for column in rows[1:]:
					m = np.vstack([m, list(map(float, column))])
				graphs["graphs"].append({"m": m, "x": "", "y": "", "label": label})
				break
			elif(xAxisTicks or yAxisTicks):
				labels = column[1] if num_columns > 1 else None
				graphs["graphs"].append({"ticks": x, "labels": labels, "label": label})
				break
			else:
				if(not (colourArea and unixTimeStamp)):
					y = [float(column[i]) if len(column[i]) > 0 else np.nan for column in rows if len(column) > i]
					#y = [float(column[i]) for column in non_str_rows if len(column) > i]
				if(generateX):
					x = range(len(y))
				graph = {"x": x, "y": y, "label": label}
				if(marker):
					graph["marker"] = markers
				if(colour):
					graph["colour"] = colours
				graphs["graphs"].append(graph)
				if(offset):
					graph["y"] = [y + cumulativeLineOffset for y in graph["y"]]
					cumulativeLineOffset += 3
				i += 1
		
		if(nolabel):
			for graph in graphs["graphs"]:
				graph["label"] = None
		return graphs
	except Exception as e:
		print(f"Failed Parsing {filename = }, {header = }")
		raise e

def main(raw_args: List[str] = None):
	shared = argparse.ArgumentParser(description = "Script to Plot the content of files.", add_help = False)
	shared.add_argument("-t", "--title", metavar = "TITLE", nargs = "*", default = [], help = "The title of the plot.")
	shared.add_argument("-o", "--outdir", metavar = "PATH", type = str, default = "", help = "The output directory for the generated pdf.")
	shared.add_argument("-i", "--interactive", action = "store_true", help = 'Shows Graph in an interactive Window.')

	parser = argparse.ArgumentParser(description = "Script to Plot the content of files.", parents = [shared])

	subparsers = parser.add_subparsers(title = "subcommands", dest = "command", required = False, description = "valid subcommands", help = "COMMAND -h for additional help.")
	
	plotCmd = subparsers.add_parser("plot", description = "Plot csv files.", parents = [shared])
	plotCmd.add_argument("paths", metavar = "PATH", nargs = "+", help = 'The csv files to plot.')

	plotlyCmd = subparsers.add_parser("plotly", description = "Plot csv files.", parents = [shared])
	plotlyCmd.add_argument("paths", metavar = "PATH", nargs = "+", help = 'The csv files to plot.')
	
	mulPlotCmd = subparsers.add_parser("multiplot", description = "Plot csv files.", parents = [shared])
	mulPlotCmd.add_argument("--ver", action = "store_true", help = 'Sets the orientation of the multiplot to vertical otherwise it will be horizontal.')
	mulPlotCmd.add_argument("-p", "--paths", metavar = "PATH", action = "append", nargs = "+", required = True, help = 'The csv files to plot.')

	mulPlotlyCmd = subparsers.add_parser("multiPlotly", description = "Plot csv files.", parents = [shared])
	mulPlotlyCmd.add_argument("--ver", action = "store_true", help = 'Sets the orientation of the multiplot to vertical otherwise it will be horizontal.')
	mulPlotlyCmd.add_argument("-p", "--paths", metavar = "PATH", action = "append", nargs = "+", required = True, help = 'The csv files to plot.')

	args = parser.parse_args(raw_args)
	try:
		print(args.paths)
		if(args.command == "plot" or args.command == ""):
			#GenerateGraphs(args.paths, outdir = args.outdir, title = args.title[0] if len(args.title) else "")
			GenerateMultiGraphs([args.paths], orientation = Orientation.VERTICAL, outdir = args.outdir, title = args.title, interactive = args.interactive)
		elif(args.command == "plotly" or args.command == ""):
			#GeneratePlotlyGraphs(args.paths, outdir = args.outdir, title = args.title[0] if len(args.title) else "")
			GenerateMultiPlotlyGraphs([args.paths], orientation = Orientation.VERTICAL, outdir = args.outdir, title = args.title, interactive = args.interactive)
		elif(args.command == "multiplot"):
			orientation = Orientation.VERTICAL if args.ver else Orientation.HORIZONTAL
			GenerateMultiGraphs(args.paths, orientation = orientation, outdir = args.outdir, title = args.title, interactive = args.interactive)
		elif(args.command == "multiPlotly"):
			orientation = Orientation.VERTICAL if args.ver else Orientation.HORIZONTAL
			GenerateMultiPlotlyGraphs(args.paths, orientation = orientation, outdir = args.outdir, title = args.title, interactive = args.interactive)
	except:
		traceback.print_exc()
		input("\nPress Anything to Exit...")

if __name__ == "__main__":
	main()
