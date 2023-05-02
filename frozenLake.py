# frozenLake.py

import numpy as np # type: ignore
import random, os, sys, glob, curses

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '../src'))

import adviceMCTS.util as util
# from mdpClasses import *
# from frozenLakeMdpClasses import *
# from simulationClasses import *

Position = Tuple[int, int]

def fileStrPos(p: Position) -> str:
	x,y=p
	return str( x ) + ',' + str( y )

def posFromFileStr(s: str) -> Position:
	split=s.split(',')
	if len(split)!=2:
		raise Exception("parsing error position "+s)
	return (int(split[0]),int(split[1]))

def actionToClass(action:str):
	if action == "North":
		a = 0
	elif action == "East":
		a = 1
	elif action == "South":
		a = 2
	elif action == "West":
		a = 3
	else:
		raise Exception("Wrong action: "+action)
	return a

def fileStrAction(action: str) -> str:
	if action in ['East','West','North','South']:
		return action[0]
	elif action=='Stop':
		return 's'
	else:
		raise Exception("unkown action "+str(action))

def actionFromFileStr(s:str) -> str:
	if s == 'N':
		return 'North'
	elif s == 'S':
		return 'South'
	elif s == 'E':
		return 'East'
	elif s == 'W':
		return 'West'
	elif s == 's':
		return 'Stop'
	else:
		raise Exception("Unknown action "+str(s))

Grid = List[List[bool]]
GridDistance = List[List[int]]

def gridCopy(grid: Grid) -> Grid:
	return [[grid[i][j] for j in range(len(grid[i]))] for i in range(len(grid))]

def gridStr(grid: Grid) -> str:
	return "\n".join([" ".join([str(grid[i][j]) for j in range(len(grid[i]))]) for i in range(len(grid))])
def wallsStr(grid: Grid) -> str:
	return "\n".join(["".join([("%" if grid[i][j] else " ") for j in range(len(grid[i]))]) for i in range(len(grid))])
def fullGridStr(walls: Grid, holes: Grid, targets: Grid) -> str:
	return "\n".join(["".join([("%" if walls[i][j] else ("_" if holes[i][j] else ("." if targets[i][j] else " "))) for j in range(len(walls[i]))]) for i in range(len(walls))])

def fullGridStrPosition(walls: Grid, holes: Grid, targets: Grid, position: Position) -> str:
	return "\n".join(["".join([("P" if (i == position[0] and j == position[1]) else ("%" if walls[i][j] else ("_" if holes[i][j] else ("." if targets[i][j] else " ")))) for j in range(len(walls[i]))]) for i in range(len(walls))])

def gridDistanceStr(grid: GridDistance) -> str:
	X=len(grid)
	Y=len(grid[0])
	infty=X*Y+1
	return "\n".join([" ".join([str(grid[i][j]) if grid[i][j] != infty else "_" for j in range(len(grid[i]))]) for i in range(len(grid))])

def wallsFromStr(s: str) -> Grid:
	grid = [sl for sl in s.split("\n")]
	return [[grid[i][j]=="%" for j in range(len(grid[i]))] for i in range(len(grid))]
def holesFromStr(s: str) -> Grid:
	grid = [sl for sl in s.split("\n")]
	return [[grid[i][j]=="_" for j in range(len(grid[i]))] for i in range(len(grid))]
def targetsFromStr(s: str) -> Grid:
	grid = [sl for sl in s.split("\n")]
	return [[grid[i][j]=="." for j in range(len(grid[i]))] for i in range(len(grid))]

def getNextPosition(position: Position, action: str) -> Position:
	(x,y)=position
	if action == 'East':
		return((x,y+1))
	elif action == 'West':
		return((x,y-1))
	elif action == 'South':
		return((x+1,y))
	elif action == 'North':
		return((x-1,y))
	elif action == 'Stop':
		return((x,y))
	else:
		raise Exception("Unknown action "+str(action))

def getLegalActions(position: Position, walls: Grid) -> List[str]:
	X=len(walls)
	Y=len(walls[0])
	x,y=position
	r=[]
	if x>0 and (not walls[x-1][y]):
		r.append('North')
	if x<X-1 and (not walls[x+1][y]):
		r.append('South')
	if y>0 and (not walls[x][y-1]):
		r.append('West')
	if y<Y-1 and (not walls[x][y+1]):
		r.append('East')
	# r.append('Stop')
	return r

def getLegalStochasticActions(position: Position, walls: Grid, action: str) -> List[str]:
	X=len(walls)
	Y=len(walls[0])
	x,y=position
	r=[]
	if x>0 and (not walls[x-1][y]) and action != 'South':
		r.append('North')
	if x<X-1 and (not walls[x+1][y]) and action != 'North':
		r.append('South')
	if y>0 and (not walls[x][y-1]) and action != 'East':
		r.append('West')
	if y<Y-1 and (not walls[x][y+1]) and action != 'West':
		r.append('East')
	# r.append('Stop')
	return r

def isLegalAction(position: Position, action: str, walls: Grid) -> bool:
	X=len(walls)
	Y=len(walls[0])
	x,y=position
	if action == 'East' and y<Y-1 and (not walls[x][y+1]):
		return True
	elif action == 'West' and y>0 and (not walls[x][y-1]):
		return True
	elif action == 'South' and x<X-1 and (not walls[x+1][y]):
		return True
	elif action == 'North' and x>0 and (not walls[x-1][y]):
		return True
	elif action == 'Stop':
		return True
	else:
		return False

def gridDistance(walls: Grid, holes: Grid, targetList: List[Position]) -> Tuple[GridDistance, int]:
	X=len(walls)
	Y=len(walls[0])
	infty=X*Y+1
	r = [[ infty for j in range(Y)] for i in range(X)]
	queue = []
	maxd=infty
	for x,y in targetList:
		r[x][y]=0
		queue.append((x,y,0))
		maxd=0
	while len(queue)>0:
		x,y,d = queue.pop(0)
		for xx,yy in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
			if xx>=0 and xx<X and yy>=0 and yy<Y and (not walls[xx][yy]) and (not holes[xx][yy]) and r[xx][yy]>d+1:
				r[xx][yy]=d+1
				queue.append((xx,yy,d+1))
				if d+1>maxd:
					maxd=d+1
	return (r,maxd)

# code to create a layout

def createWalls(height,width,p = 0.9):
	grid = []
	for i in range(height):
		row = []
		for j in range(width):
			r = random.random()
			if i == 0 or i == height-1 or j == 0 or j == width -1:
				row.append(True)
			elif r > p:
				row.append(True)
			else:
				row.append(False)
		grid.append(row)
	return grid

def addOther(walls, p = 0.9):
	height = len(walls)
	width = len(walls[0])
	# holes = [[False for j in range(width)] for i in range(height)]
	targets = [[False for j in range(width)] for i in range(height)]
	holes = []
	for i in range(height):
		row = []
		for j in range(width):
			r = random.random()
			if walls[i][j]:
				row.append(False)
			elif r > p:
				row.append(True)
			else:
				row.append(False)
		holes.append(row)
	while True:
		tx = random.randint(0,height-1)
		ty = random.randint(0,width-1)
		if not walls[tx][ty] and not holes[tx][ty]:
			targets[tx][ty] = True
			break
		else:
			pass
	while True:
		px = random.randint(0,height-1)
		py = random.randint(0,width-1)
		if not walls[px][py] and not holes[px][py] and not targets[px][py]:
			position = (px,py)
			break
		else:
			pass
	return walls,holes,targets,position

def fileStr(walls,holes,targets,position,drawHorizon,discount = 1):
	s = fullGridStr(walls,holes,targets)+'\nParameters\ndraw '+str(drawHorizon)+'\ndiscount '+str(discount)+'\nInitPosition\n'+fileStrPos(position)+'\nInitPredicates\n'
	x,y = position
	predicates = []
	if holes[x][y]:
		predicates.append("Loss")
	if targets[x][y]:
		predicates.append("Win")
	s += ' '.join(predicates)
	return s

def createLayouts (n,height,width,layouts_dir,p=0.9,prefix = ''): # bigger value of p = easier layouts
	util.mkdir(layouts_dir)
	fileList = []
	fname = layouts_dir+os.sep+prefix+'_'+str(height)+'x'+str(width)+'_'
	for i in range(n):
		f = open(fname+str(i)+'.lay','w+')
		walls = createWalls(height,width,p=p)
		walls,holes,targets,position = addOther(walls,p=p)
		drawHorizon = 2*(height+width)
		s = fileStr(walls,holes,targets,position,drawHorizon)
		print(s,file=f)
		# f.write(s)
		f.close()
		fileList.append(fname+str(i)+'.lay')
	return fileList

def createRandomGrid (n,height,width,p=0.9,prefix = ''): # bigger value of p = easier layouts
	for i in range(n):
		walls = createWalls(height,width,p=p)
		walls,holes,targets,position = addOther(walls,p=p)
	return walls,holes,targets,position

Grid = List[List[bool]]

def gridToArray(grid:Grid): # creates numpy arrays from grids
	X = len(grid)
	Y=len(grid[0])
	array = np.zeros((1,X,Y))
	for i in range(X):
		for j in range(Y):
			if grid[i][j]:
				array[0][i][j] = 1
	return array

def addPositionToArray(array, position):
	x,y = position
	_,X,Y = array.shape
	pArray = np.zeros((1,X,Y))
	pArray[0][x][y] = 1
	return np.concatenate((array , pArray), axis = 0)

def addActionToArray(array, action:str):
	_,X,Y = array.shape
	aArray = np.ones((1,X,Y))
	if action == "North":
		aArray *= 0.25
	elif action == "East":
		aArray *= 0.5
	elif action == "South":
		aArray *= 0.75
	elif action == "West":
		aArray *= 1
	else:
		raise Exception ("Unknown Action: "+action)
	return np.concatenate((array , aArray), axis = 0)

def gridsToArray(walls, holes, targets):
	wallArray = gridToArray(walls)
	holeArray = gridToArray(holes)
	targetArray = gridToArray(targets)
	layoutArray =  np.concatenate((wallArray , holeArray, targetArray), axis = 0)
	return layoutArray

def gridsToArrayWithPos(walls, holes, targets, position):
	layoutArray = gridsToArray(walls, holes, targets)
	return addPositionToArray(layoutArray, position)


def distanceM(pos1,pos2): #Manhattan distance
	(x1,y1)=pos1
	(x2,y2)=pos2
	return abs(x1-x2)+abs(y1-y2)

if __name__ == '__main__':
	createLayouts (5,7,10,"test",p=0.9,prefix = '')
	pass
