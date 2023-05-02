# gridMdpClasses.py

# from __future__ import annotations

# Positions in a grid with height X and width Y :
# (0,0)    (0,1)  . . .  (0,Y-1)
# (1,0)    (1,1)            .
# .               .         .
# .                 .       .
# .                   .     .
# (X-1,0) (X-1,1)...(X-1,Y-1)

import random, time, curses, os, sys
import numpy as np
from tensorflow import keras
from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn
from frozenLakeStorm import *

# CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(CURRENTPWD, '../src'))

import adviceMCTS.util as util
from adviceMCTS.mdpClasses import *
from frozenLake import *
from adviceMCTS.simulationClasses import *

class MDPPredicate(MDPPredicateInterface):
	# methods that must be redefined
	def __init__(self,name: str):
		self.name = name
	def deepCopy(self) -> "MDPPredicate":
		return MDPPredicate(self.name)
	def initFromCopy(self, other: "MDPPredicate") -> None:
		self.name = other.name
	def __str__(self) -> str:
		return "(name:"+str(self.name)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.name)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return str(self.name)
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPPredicate":
		# if s=="":
			# raise Exception("parsing error") ## NOTE: why this?
		name=s
		return cls(name)

class MDPState(MDPStateInterface):
	# methods that must be redefined
	def __init__(self,position: Position) -> None:
		self.position=position # a pair (x,y) of integers
	def deepCopy(self) -> "MDPState":
		return MDPState(self.position)
	def initFromCopy(self, other: "MDPState") -> None:
		self.position = other.position
	def __str__(self) -> str:
		return "(position:"+str(self.position)+")"
	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPState):
			return NotImplemented
		return (self.position == other.position)

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.position[0])+','+str(self.position[1])

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return fileStrPos(self.position)
	@classmethod
	def fromFileStr(cls, s:str) -> "MDPState":
		if s=="":
			raise Exception("parsing error")
		position=posFromFileStr(s)
		return cls(position)

class MDPAction(MDPActionInterface):
	# Immutable class
	# methods that must be redefined
	def __init__(self,action: str,infoStr: str) -> None:
		self.action=action
		self.infoStr=infoStr
	def deepCopy(self) -> "MDPAction":
		return MDPAction(self.action,self.infoStr)
	def __hash__(self) -> int:
		return hash(self.action)
	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPAction):
			return NotImplemented
		return (self.action == other.action)
	def __str__(self) -> str:
		return "(action:"+str(self.action)+",infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.action)+("#"+str(self.infoStr) if self.infoStr!="" else "")
	def miniConsoleStr(self) -> str:
		return str(self.action)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return fileStrAction(self.action)+("\n#"+str(self.infoStr) if self.infoStr!="" else "")
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPAction":
		if s=="":
			raise Exception("parsing error")
		splits=s.split("\n#")
		action=actionFromFileStr(splits[0])
		infoStr=""
		for i in range(1,len(splits)):
			infoStr+=splits[i]
		return cls(action,infoStr)


class MDPStochasticAction(MDPStochasticActionInterface):
	# Immutable class
	# methods that must be redefined
	def __init__(self,action: str,infoStr: str) -> None:
		self.action=action
		self.infoStr=infoStr
	def deepCopy(self) -> "MDPStochasticAction":
		return MDPStochasticAction(self.action,self.infoStr)
	def __hash__(self) -> int:
		return hash(self.action)
	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPStochasticAction):
			return NotImplemented
		return (self.action == other.action)
	def __str__(self) -> str:
		return "(action:"+str(self.action)+",infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.action)+("#"+str(self.infoStr) if self.infoStr!="" else "")
	def miniConsoleStr(self) -> str:
		return str(self.action)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return fileStrAction(self.action)+("\n#"+str(self.infoStr) if self.infoStr!="" else "")
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPStochasticAction":
		if s=="":
			raise Exception("parsing error")
		splits=s.split("\n#")
		action=actionFromFileStr(splits[0])
		infoStr=""
		for i in range(1,len(splits)):
			infoStr+=splits[i]
		return cls(action,infoStr)

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

class MDPOperations(MDPOperationsInterface[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]):
	FILE_SEPARATOR: str = "\nParameters\n"

	# methods that must be redefined
	def __init__(self, walls: Grid, holes: Grid, targets: Grid, drawHorizon: int, discountFactor: float) -> None:
		self.walls=walls
		self.holes=holes
		self.targets=targets
		self.drawHorizon=drawHorizon
		self.discountFactor=discountFactor

		self.targetList=[]
		self.holeList=[]
		for i in range(len(targets)):
			for j in range(len(targets[i])):
				if targets[i][j]:
					self.targetList.append((i,j))
		for i in range(len(holes)):
			for j in range(len(holes[i])):
				if holes[i][j]:
					self.holeList.append((i,j))
		self.holeDistance,self.maxHoleDistance = gridDistance(walls,targets,self.holeList)
		self.targetDistance,self.maxTargetDistance = gridDistance(walls,holes,self.targetList)
		pArray = np.zeros((1,len(walls),len(walls[0])))
		holeDistanceTable = np.zeros((1,len(walls),len(walls[0])))
		for i in range(len(walls)):
			for j in range(len(walls[0])):
				holeDistanceTable[0][i][j] = normalizeDistance(self.holeDistance[i][j],self.maxHoleDistance)
		targetDistanceTable = np.zeros((1,len(walls),len(walls[0])))
		for i in range(len(walls)):
			for j in range(len(walls[0])):
				targetDistanceTable[0][i][j] = normalizeDistance(self.targetDistance[i][j],self.maxTargetDistance)
		self.config = np.concatenate((gridToArray(walls),gridToArray(holes),gridToArray(targets),pArray,holeDistanceTable,targetDistanceTable),axis=0)

	def deepCopy(self) -> "MDPOperations":
		return MDPOperations(gridCopy(self.walls),gridCopy(self.holes),gridCopy(self.targets),self.drawHorizon,self.discountFactor)
	def __str__(self) -> str:
		return "(walls:\n"+gridStr(self.walls)+",holes:\n"+gridStr(self.holes)+",targets:\n"+gridStr(self.targets)+",discountFactor:"+str(self.discountFactor)+")"

	def getConfig(self, mdpState: MDPState):
		position = mdpState.position
		config = np.copy(self.config)
		config[3][position[0]][position[1]] = 1
		return(config)

	def applyTransitionOnState(self, mdpState: MDPState, mdpTransition: MDPTransition[MDPAction, MDPStochasticAction]) -> float:
		if not isLegalAction(mdpState.position,mdpTransition.mdpAction.action,self.walls):
			raise Exception("illegal action played with "+str(mdpTransition)+" from "+str(mdpState))
		newPos=getNextPosition(mdpState.position,mdpTransition.mdpStochasticAction.action)
		mdpState.position=newPos
		mdpReward=-1
		return mdpReward
	def getStochasticActions(self, mdpState: MDPState, mdpAction: MDPAction, quietInfoStr: bool) -> MDPStochasticAction:
		mdpStochasticActions = [MDPStochasticAction(label,"") for label in getLegalStochasticActions(mdpState.position,self.walls,mdpAction.action)]
		# print("trying to play",mdpAction,"returns"," ".join([a.action for a in mdpStochasticActions]))
		return mdpStochasticActions

	def getDistribution(self, mdpState: MDPState, mdpAction: MDPAction):
		mdpStochasticActions = [MDPStochasticAction(label,"") for label in getLegalStochasticActions(mdpState.position,self.walls,mdpAction.action)]
		# print("trying to play",mdpAction,"returns"," ".join([a.action for a in mdpStochasticActions]))
		dist: util.ConsoleStrFloatCounter[MDPStochasticAction] = util.ConsoleStrFloatCounter()
		for a in mdpStochasticActions:
			if a.action == mdpAction.action:
				dist[a] = 10.0
			else:
				dist[a] = 1.0
		dist.normalize()
		return dist

	def drawStochasticAction(self, mdpState: MDPState, mdpAction: MDPAction, quietInfoStr: bool) -> MDPStochasticAction:
		mdpStochasticActions = [MDPStochasticAction(label,"") for label in getLegalStochasticActions(mdpState.position,self.walls,mdpAction.action)]
		# print("trying to play",mdpAction,"returns"," ".join([a.action for a in mdpStochasticActions]))
		dist: util.ConsoleStrFloatCounter[MDPStochasticAction] = util.ConsoleStrFloatCounter()
		for a in mdpStochasticActions:
			if a.action == mdpAction.action:
				dist[a] = 10.0
			else:
				dist[a] = 1.0
		dist.normalize()
		if len(dist) == 0:
			# return None
			raise Exception("empty distribution")
		choice = util.chooseFromDistribution( dist ).deepCopy()
		if not quietInfoStr:
			isUniversal = True
			vv = None
			for k,v in dist.items():
				if vv is None : vv = v
				if v != vv: isUniversal = False
			if isUniversal:
				if choice.infoStr != '':
					choice.infoStr += '#'
				choice.infoStr += 'DistUniversal'
			else:
				# pass
				if choice.infoStr != '':
					choice.infoStr += '#'
				choice.infoStr += 'Dist'+str(dist)
		return choice
	def getLegalActions(self, mdpState: MDPState) -> List[MDPAction]:
		legalActions=[]
		for label in getLegalActions(mdpState.position,self.walls):
			legalActions.append(MDPAction(label,""))
		return legalActions

	# methods that can be redefined
	def consoleStr(self) -> str:
		return "{grid:\n"+fullGridStr(self.walls,self.holes,self.targets)+discountFactorStr(self.discountFactor)+"}"
	def replayConsoleStr(self, mdpState: MDPState) -> str:
		return fullGridStrPosition(self.walls,self.holes,self.targets,mdpState.position)
	def getAllPredicates(self) -> List[MDPPredicate]:
		mdpPredicates: List[MDPPredicate] = [MDPPredicate("Win"), MDPPredicate("Loss")] # ,MDPPredicate("Draw")
		# list all predicates available, true or false
		# mdpPredicates.append(MDPPredicate())
		return mdpPredicates
	def getPredicates(self, mdpState: MDPState) -> List[MDPPredicate]:
		mdpPredicates: List[MDPPredicate] = []
		x,y = mdpState.position
		if self.holes[x][y]:
			mdpPredicates.append(MDPPredicate("Loss"))
		if self.targets[x][y]:
			mdpPredicates.append(MDPPredicate("Win"))
		# mdpPredicates.append(MDPPredicate(fileStrPos((x,y))))
		# list all predicates that hold on mdpState
		# mdpPredicates.append(MDPPredicate())
		return mdpPredicates
	def isExecutionTerminal(self, mdpExecution: MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]) -> bool:
		# Redefine this method if you want a notion of terminal state/path
		x,y = mdpExecution.mdpEndState.position
		if self.holes[x][y] or self.targets[x][y] or self.getLegalActions(mdpExecution.mdpEndState) == []:# or mdpExecution.length() == self.drawHorizon
			return True
		# is the execution over? (terminal state reached, horizon reached, etc)
		return False
	def getTerminalReward(self, mdpExecution: MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]) -> float:
		# Redefine this method if you want a notion of terminal reward
		# penalty for losing for exemple
		mdpTerminalReward=0.0
		x,y = mdpExecution.mdpEndState.position
		if self.targets[x][y]:
			mdpTerminalReward=100.0
		elif self.holes[x][y] or self.getLegalActions(mdpExecution.mdpEndState) == []:
			mdpTerminalReward=-100.0
		# elif mdpExecution.length() == self.drawHorizon:
		# 	mdpTerminalReward=0
		else:
			mdpTerminalReward=0
			# raise Exception("execustion not Terminal")
		# reward for terminal states
		return mdpTerminalReward * mdpExecution.discountFactor

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return fullGridStr(self.walls,self.holes,self.targets)+MDPOperations.FILE_SEPARATOR+"draw"+str(self.drawHorizon)+"\ndiscount"+str(self.discountFactor)
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPOperations":
		split=s.split(MDPOperations.FILE_SEPARATOR)
		if len(split)!=2:
			raise Exception("parsing error mdp "+s)
		walls=wallsFromStr(split[0])
		holes=holesFromStr(split[0])
		targets=targetsFromStr(split[0])
		params = split[1].split('\n')
		if params[0][:4]!="draw":
			raise Exception("bad str provided",params[0])
		drawHorizon=int(params[0][4:])
		if params[1][:8]!="discount":
			raise Exception("bad str provided",params[1])
		discountFactor=float(params[1][8:])
		return cls(walls,holes,targets,drawHorizon,discountFactor)

# Parsers for simulation classes

def MDPTransitionfromFileStr(s:str) -> MDPTransition[MDPAction,MDPStochasticAction]:
	splits=s.split(MDPTransition.FILE_SEPARATOR)
	s1=splits[0]
	s2=""
	for i in range(1,len(splits)):
		s2+=splits[i]
	return MDPTransition(MDPAction.fromFileStr(s1),MDPStochasticAction.fromFileStr(s2))

def MDPPathfromFileStr(s:str) -> MDPPath[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]:
	if MDPPath.FILE_PREFIX==s[:len(MDPPath.FILE_PREFIX)]:
		s=s[len(MDPPath.FILE_PREFIX):]
	else:
		raise Exception("bad prefix in parsing path")
	splits=s.split(MDPPath.FILE_SEPARATOR)
	s1=splits[0]
	s2=""
	for i in range(1,len(splits)):
		s2+=splits[i]
	splits1 = s2.split(MDPPath.FILE_PREDICATE_SEPARATOR)
	if len(splits1) != 2:
		raise Exception("Parse error")
	splits2=splits1[0].split(MDPPath.FILE_LIST_SEPARATOR)
	splits3=splits1[1].split(MDPPath.FILE_LIST_SEPARATOR)
	return MDPPath(MDPState.fromFileStr(s1),[MDPTransitionfromFileStr(ss) for ss in splits2],[[MDPPredicate.fromFileStr(sss) for sss in ss.split(" ")] for ss in splits3])

def MDPExecutionfromFileStr(s:str) -> MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]:
	splits=s.split(MDPExecution.FILE_SEPARATOR1)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpPath = MDPPathfromFileStr(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR2)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpEndState = MDPState.fromFileStr(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR3)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpPathReward = float(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR4)
	if len(splits) != 2 or not (splits[0] == "True" or splits[0] == "False"):
		raise Exception("Parse error")
	isTerminal = (splits[0] == "True")
	discountFactor = float(splits[1])
	return MDPExecution(mdpPath, mdpEndState, mdpPathReward, isTerminal, discountFactor)

def readResults(file: Any) -> List[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]]:
	FILE_PREFIX = 'MDP:\n'
	s = file.read()
	s = s.split('\nTraceLists:\n')
	if len(s) != 2:
		raise Exception("Parsing error")
	so = s[0]
	if FILE_PREFIX==so[:len(FILE_PREFIX)]:
		so = so[len(FILE_PREFIX):]
	else:
		raise Exception("bad prefix in parsing path")
	mdpOperations = MDPOperations.fromFileStr(so) # type: Any
	st = s[1]
	TRACE_SEPERATOR = '\nTrace:\n'
	st = st.split(TRACE_SEPERATOR)
	engineList = []
	for t in st:
		st = t.split('\nnumSim:\n')
		if len(st) != 2:
			print(st)
			raise Exception("parser errror")
		mdpExecution = MDPExecutionfromFileStr(st[0])
		numSim = int(st[1])
		if numSim != 1:
			raise Exception("numSim not 1")
		engine = MDPExecutionEngine(mdpOperations, mdpExecution) # type: MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]
		engineList.append(engine)
	return engineList

# Save to file

def printResults(traces: List[Tuple[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction],float,int]], file: Any = sys.stdout) -> None:
	TRACE_SEPERATOR = '\nTrace:\n'
	if len(traces)<1 or len(traces[0])<1:
		raise Exception("empty trace")
	print ('MDP:\n'+traces[0][0].mdpOperations.fileStr()+'\nTraceLists:\n'+TRACE_SEPERATOR.join([r[0].mdpExecution.fileStr()+'\nnumSim:\n'+str(r[2]) for r in traces]),file=file)


def runResults(engineList: List[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]], quiet: bool = True, prettyConsole: bool = False) -> Any:
	print("[== running replay engines")
	for mdpExecutionEngine in engineList:
		cursesScr = None
		cursesDelay = 0.0
		if prettyConsole:
			cursesScr = curses.initscr()
			curses.noecho()
			curses.cbreak()
			cursesDelay = 0.5#min(0.5,max(0.05,10.0/mdpExecutionEngine.length(ignoreNonDecisionStates=True)))

		optionsReplayEngine=OptionsReplayEngine(quiet=quiet, printEachStep=False, printCompact=False, cursesScr=cursesScr, cursesDelay=cursesDelay)
		mdpReplayEngine = MDPReplayEngine(mdpExecutionEngine.mdpOperations,mdpExecutionEngine.mdpPath(),options=optionsReplayEngine)
		try:
			while not mdpReplayEngine.isTerminal():
				mdpReplayEngine.advanceReplay()
				# print("step",i,":",mdpReplayEngine.mdpEndState().consoleStr())

			if (mdpReplayEngine.mdpExecutionEngine.isTerminal() != mdpExecutionEngine.isTerminal()) or (mdpReplayEngine.mdpExecutionEngine.mdpPathReward() != mdpExecutionEngine.mdpPathReward()) or (mdpReplayEngine.mdpExecutionEngine.mdpEndState() != mdpExecutionEngine.mdpEndState()):
				print (mdpReplayEngine.mdpExecutionEngine.isTerminal())
				print (mdpExecutionEngine.isTerminal())
				print (mdpReplayEngine.mdpExecutionEngine.mdpPathReward())
				print (mdpExecutionEngine.mdpPathReward())
				print (mdpReplayEngine.mdpExecutionEngine.mdpEndState())
				print (mdpExecutionEngine.mdpEndState())
				raise Exception("bad replay")
			# print("replay done")
			if prettyConsole:
				curses.endwin()
		except BaseException as error:
			if prettyConsole:
				curses.endwin()
			raise Exception(str(error))


		# mdpReplayEngine.resetReplay()
	print("==] done")

# Read a layout
def readFromFile(fname):
	f = open(fname)
	FILE_SEPARATOR1="\nInitPredicates\n"
	FILE_SEPARATOR2="\nInitPosition\n"
	s = f.read()
	f.close()
	s = s.split(FILE_SEPARATOR1)
	if len(s) != 2:
		raise Exception("bad layout" + str(s))
	s0 = s[0].split(FILE_SEPARATOR2)
	if len(s0) != 2:
		raise Exception("bad layout")
	initPredicates = [MDPPredicate.fromFileStr(t) for t in s[1].split(' ')]
	mdp = MDPOperations.fromFileStr(s0[0])
	initState = MDPState.fromFileStr(s0[1])
	return mdp,initState,initPredicates

def gridsFromFile(fname):
	f = open(fname)
	FILE_SEPARATOR="\nInitPredicates\n"
	s = f.read().split(FILE_SEPARATOR)[0]
	FILE_SEPARATOR="\nInitPosition\n"
	s = s.split(FILE_SEPARATOR)
	f.close()
	mdpDescription = s[0]
	position = posFromFileStr(s[1])
	FILE_SEPARATOR="\nParameters\n"
	mdpDescription = mdpDescription.split(FILE_SEPARATOR)
	walls=wallsFromStr(mdpDescription[0])
	holes=holesFromStr(mdpDescription[0])
	targets=targetsFromStr(mdpDescription[0])
	return (walls, holes, targets, position)

def mdpFromGrids(walls, holes, targets, position):
	mdp = MDPOperations(walls= walls, holes = holes, targets = targets, drawHorizon = 1000, discountFactor = 1)
	initState = MDPState(position = position)
	initPredicates = []
	return mdp,initState,initPredicates

def runGames(**kwargs):
	layout = kwargs['layout']
	kwargs.pop('layout')
	mdp, initState, initPredicates = readFromFile(layout)
	isReplay = kwargs['replay']
	kwargs.pop('replay')
	if kwargs['useMCTS']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPMCTSTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPMCTSTraceEngine()
		results = traceEngine.runMCTSTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	elif kwargs['useDT']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPDTTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPDTTraceEngine()
		results = traceEngine.runDTTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	elif kwargs['useNN']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPNNTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPNNTraceEngine()
		results = traceEngine.runNNTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	elif kwargs['useMultiNN']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPMultiNNTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPMultiNNTraceEngine()
		results = traceEngine.runMultiNNTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	elif kwargs['useStorm']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPStormTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPStormTraceEngine()
		results = traceEngine.runStormTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	elif kwargs['useStormDist']:
		kwargs.pop('useMCTS')
		kwargs.pop('useDT')
		kwargs.pop('useNN')
		kwargs.pop('useMultiNN')
		kwargs.pop('useStorm')
		kwargs.pop('useStormDist')
		traceEngine: MDPStormDistTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPStormDistTraceEngine()
		results = traceEngine.runStormDistTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp, **kwargs)
	if isReplay:
		engineList = [r[0] for r in results]
		runResults(engineList,quiet=True,prettyConsole=True)
	return results

class MDPStateScoreSimple(MDPStateScoreInterface):
	"""
	Score using Manhattan distance
	"""
	def getScore(self, executionEngine):
		position = executionEngine.mdpEndState().position
		walls = executionEngine.mdpOperations.walls
		holes = executionEngine.mdpOperations.holes
		targets = executionEngine.mdpOperations.targets
		score = 0
		for i in range(len(targets)):
			for j in range(len(targets[i])):
				if targets[i][j]:
					posT=(i,j)
					d = distanceM(position,posT)
					score += d
		for i in range(len(holes)):
			for j in range(len(holes[i])):
				if holes[i][j]:
					posT=(i,j)
					d = distanceM(position,posT)
					if d==0:
						score += 10
		return -score

class MDPStateScoreFast(MDPStateScoreInterface):
	def getScore(self, executionEngine):
		position = executionEngine.mdpEndState().position
		scoreTbl = [0.0,0.0,0.0,0.0]
		score = 0.0
		X=len(executionEngine.mdpOperations.walls)
		Y=len(executionEngine.mdpOperations.walls[0])
		dm=X+Y+1
		dM=0
		for posT in executionEngine.mdpOperations.targetList:
			d = distanceM(position,posT)
			if d>dM:
				dM=d
			if d<dm:
				dm=d
		if dm == 0:
			score += 1
			scoreTbl[0] = 1
		else:
			score += -dm/(X+Y+1)
			scoreTbl[1] = -dm/(X+Y+1)
		dm=X+Y+1
		dM=0
		for posH in executionEngine.mdpOperations.holeList:
			d = distanceM(position,posH)
			if d>dM:
				dM=d
			if d<dm:
				dm=d
		if dm == 0:
			score += -1
			scoreTbl[2] = -1
		else:
			score += dm/(X+Y+1)
			scoreTbl[3] = dm/(X+Y+1)

		if scoreTbl[0] != 0 or scoreTbl[2] != 0:
			finalScore = scoreTbl[0] + scoreTbl[2]
		else:
			finalScore = (scoreTbl[1]+scoreTbl[3])
		finalScore = (finalScore+1)/2
		# print(executionEngine.mdpOperations.replayConsoleStr(executionEngine.mdpEndState()))
		# print(scoreTbl,":",finalScore)
		return finalScore

def normalizeFloat(f,minf,maxf):
	if minf>maxf:
		raise Exception(f"bad normalisation: {minf}, {maxf}")
	elif minf == maxf:
		return 0.0
	f = max(f,minf)
	f = min(f,maxf)
	return (f-minf)/(maxf-minf)

def normalizeDistance(d,maxd):
	return normalizeFloat(d,0,maxd)

class MDPStateScoreDistance(MDPStateScoreInterface):
	def getScore(self, executionEngine):
		x,y = executionEngine.mdpEndState().position
		if executionEngine.mdpOperations.targets[x][y]:
			return 1.0
		if executionEngine.mdpOperations.holes[x][y]:
			return 0.0
		targetScore=1-normalizeDistance(executionEngine.mdpOperations.targetDistance[x][y],executionEngine.mdpOperations.maxTargetDistance)
		holeScore=normalizeDistance(executionEngine.mdpOperations.holeDistance[x][y],executionEngine.mdpOperations.maxHoleDistance)
		targetWt=9
		holeWt=1
		return normalizeFloat(targetWt*targetScore + holeWt*holeScore,0,targetWt+holeWt)

class MDPStateScoreNN(MDPStateScoreInterface):

	def __init__(self,model,layout):
		self.model = model
		self.distanceScore=MDPStateScoreDistance()
		mdp,initState,initPredicates = readFromFile(layout)
		walls = mdp.walls
		holes=mdp.holes
		targets=mdp.targets
		pArray = np.zeros((1,len(walls),len(walls[0])))
		holeDistanceTable = np.zeros((1,len(walls),len(walls[0])))
		for i in range(len(walls)):
			for j in range(len(walls[0])):
				holeDistanceTable[0][i][j] = normalizeDistance(mdp.holeDistance[i][j],mdp.maxHoleDistance)
		targetDistanceTable = np.zeros((1,len(walls),len(walls[0])))
		for i in range(len(walls)):
			for j in range(len(walls[0])):
				targetDistanceTable[0][i][j] = normalizeDistance(mdp.targetDistance[i][j],mdp.maxTargetDistance)
		self.array = np.expand_dims(np.concatenate((walls,holes,targets,pArray,holeDistanceTable,targetDistanceTable),axis=0),axis=0)

	def getScore(self, executionEngine):
		position = executionEngine.mdpEndState().position
		if executionEngine.mdpOperations.targets[position[0]][position[1]]:
			return 1.0
		if executionEngine.mdpOperations.holes[position[0]][position[1]]:
			return 0.0
		x = np.copy(self.array)
		x[0][3][position[0]][position[1]] = 1
		y = self.model(x)[0][0]
		return 0.9*normalizeFloat(y,0,1)+0.1*self.distanceScore.getScore(executionEngine)

class MDPStateScore(MDPStateScoreDistance):
	pass


class MDPNonLossPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		mdpPredicatesList = mdpExecutionEngine.mdpOperations.getPredicates(mdpExecutionEngine.mdpEndState())
		for predicate in mdpPredicatesList:
			if predicate.name == "Loss":
				return False
		return True

class MDPEXNonLossActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	def _isMDPActionAllowed(self, mdpAction: TMDPAction, mdpState: TMDPState, mdpOperations: TMDPOperations) -> bool:
		mdpStochasticActions = mdpOperations.getStochasticActions(mdpState, mdpAction)
		r = False
		for mdpStochasticAction in mdpStochasticActions:
			mdpSate0=mdpState.getFastResetData()
			mdpTransition=MDPTransition(mdpAction,mdpStochasticAction)
			mdpOperations.applyTransitionOnState(mdpState,mdpTransition)
			mdpPredicatesList = mdpOperations.getPredicates(mdpState)
			mdpState.fastReset(mdpSate0)
			for predicate in mdpPredicatesList:
				if predicate.name != "Loss":
					r = True
					break
		return r

class MDPAXNonLossActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	def _isMDPActionAllowed(self, mdpAction: TMDPAction, mdpState: TMDPState, mdpOperations: TMDPOperations) -> bool:
		mdpStochasticActions = mdpOperations.getStochasticActions(mdpState, mdpAction)
		r = True
		for mdpStochasticAction in mdpStochasticActions:
			mdpSate0=mdpState.getFastResetData()
			mdpTransition=MDPTransition(mdpAction,mdpStochasticAction)
			mdpOperations.applyTransitionOnState(mdpState,mdpTransition)
			mdpPredicatesList = mdpOperations.getPredicates(mdpState)
			mdpState.fastReset(mdpSate0)
			for predicate in mdpPredicatesList:
				if predicate.name == "Loss":
					r = False
					break
		return r

class MDPDTActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A advice that gives a legal action according to a decision tree.
	"""

	def __init__(self,tree):
		self.tree = tree

	def deepCopy(self):
		return MDPDTActionAdvice(tree=self.tree)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		stateConfig = np.transpose(mdpOperations.getConfig(mdpState), (1, 2, 0))
		configShape = stateConfig.shape
		flattenedConfig = np.reshape(stateConfig, (1, configShape[0] * configShape[1] * configShape[2]))
		actionName = self.tree.predict(flattenedConfig)[0]
		for action in mdpActions:
			if action.action == actionName:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		return choices

class MDPNNActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A advice that gives a legal action according to a neural network.
	"""

	def __init__(self,model, threshold):
		self.model = model
		self.threshold = threshold

	def deepCopy(self):
		return MDPNNActionAdvice(model=self.model, threshold=self.threshold)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		output_shape = self.model.layers[-1].output_shape
		if output_shape[1] == 4:
			x = np.expand_dims(mdpOperations.getConfig(mdpState)[:4],axis=0)
			# print(mdpOperations.replayConsoleStr(mdpState))
			# print(np.shape(x))
			x = np.transpose(x, [0, 2, 3, 1]) # converting CHW to HWC
			actionValues = keras.backend.get_value(self.model(x))[0]
			# print(actionValues)
			maxValue = max(actionValues)
			for action in mdpActions:
				actionId = ['East','West','North','South'].index(action.action)
				if actionValues[actionId] >= self.threshold*maxValue:
					# print(actionName)
					choices.append(action)

		else:
			raise Exception("wrong shape")
		if len(choices)==0:
			return mdpActions
		return choices

class MDPMultiNNActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A advice that gives a legal action according to neural networks.
	"""

	def __init__(self,model1, threshold1,model2, threshold2):
		self.model1 = model1
		self.threshold1 = threshold1
		self.model2 = model2
		self.threshold2 = threshold2
		self.mdpNNActionAdvice1 = MDPNNActionAdvice(model1, threshold1)
		self.mdpNNActionAdvice2 = MDPNNActionAdvice(model2, threshold2)

	def deepCopy(self):
		return MDPNNActionAdvice(model1=self.model1, threshold1=self.threshold1,model2=self.model2, threshold2=self.threshold2)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices1 = self.mdpNNActionAdvice1._getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)
		if len(choices1)==0:
			return mdpActions
		# print("2nd advice")
		choices2 = self.mdpNNActionAdvice2._getMDPActionAdviceInSubset(choices1, mdpState, mdpOperations)
		if len(choices2)==0:
			return choices1
		return choices2

class MDPStormActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A advice that gives a legal action according to a neural network.
	"""

	def __init__(self, threshold):
		self.threshold = threshold

	def deepCopy(self):
		return MDPStormActionAdvice(threshold=self.threshold)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		walls = mdpOperations.walls
		holes = mdpOperations.holes
		targets = mdpOperations.targets
		position = mdpState.position
		height = len(walls)
		width = len(walls[0])

		actionValues = getAllValuesFromGrids(walls,holes,targets,position)
		# print(mdpOperations.replayConsoleStr(mdpState))
		# print(actionValues)

		maxValue = max(actionValues)
		for action in mdpActions:
			actionId = ['East','West','North','South'].index(action.action)
			if actionValues[actionId] >= self.threshold*maxValue:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		return choices

class MDPStormDistActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):

	def __init__(self):
		pass

	def deepCopy(self):
		return MDPStormActionAdvice()

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		walls = mdpOperations.walls
		holes = mdpOperations.holes
		targets = mdpOperations.targets
		position = mdpState.position
		height = len(walls)
		width = len(walls[0])

		actionValues = getAllDistValuesFromGrids(walls,holes,targets,position)
		# print(mdpOperations.replayConsoleStr(mdpState))
		# print(actionValues)

		# maximize probability
		maxValue = max([actionValue[0] for actionValue in actionValues])
		for action in mdpActions:
			actionId = ['East','West','North','South'].index(action.action)
			if actionValues[actionId][0] >= maxValue - 0.001:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions

		# maximize distance
		optChoices = []
		minValue = min([actionValue[1] for actionValue in actionValues])
		for action in choices:
			actionId = ['East','West','North','South'].index(action.action)
			if actionValues[actionId][1] <= minValue:
				# print(actionName)
				optChoices.append(action)
		if len(optChoices)==0:
			return choices
		# print([action.action for action in optChoices])
		return optChoices

class MDPDictActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):

	def __init__(self,d):
		self.d = d

	def deepCopy(self):
		return MDPDictActionAdvice(self.d)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		position = mdpState.position

		actionValues = self.d[position]
		# print(mdpOperations.replayConsoleStr(mdpState))
		# print(actionValues)

		# maximize probability
		maxValue = max([actionValue[0] for actionValue in actionValues])
		for action in mdpActions:
			actionId = ['East','West','North','South'].index(action.action)
			if actionValues[actionId][0] >= maxValue:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions

		# maximize distance
		optChoices = []
		minValue = min([actionValue[1] for actionValue in actionValues])
		for action in choices:
			actionId = ['East','West','North','South'].index(action.action)
			if actionValues[actionId][1] <= minValue:
				# print(actionName)
				optChoices.append(action)
		if len(optChoices)==0:
			return choices
		# print([action.action for action in optChoices])
		return optChoices


class MDPDTActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to a decision tree.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, tree) -> None:
		self.tree = tree
		self.advice = MDPDTActionAdvice(tree)

	def deepCopy(self) -> "MDPDTActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPDTActionStrategy(tree=self.tree)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPNNActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to a neural network.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, model, threshold) -> None:
		self.model = model
		self.advice = MDPNNActionAdvice(model,threshold)
		self.threshold = threshold

	def deepCopy(self) -> "MDPNNActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPNNActionStrategy(model=self.model,threshold=self.threshold)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPMultiNNActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to neural networks.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, model1, threshold1, model2, threshold2) -> None:
		self.model1 = model1
		self.threshold1 = threshold1
		self.model2 = model2
		self.threshold2 = threshold2
		self.advice = MDPMultiNNActionAdvice(model1,threshold1,model2,threshold2)

	def deepCopy(self) -> "MDPNNActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPNNActionStrategy(model1=self.model1,threshold1=self.threshold1,model2=self.model2,threshold2=self.threshold2)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPStormActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to storm.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, threshold) -> None:
		self.advice = MDPStormActionAdvice(threshold)
		self.threshold = threshold

	def deepCopy(self) -> "MDPStormActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPStormActionStrategy(threshold=self.threshold)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPStormDistActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to storm.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self) -> None:
		self.advice = MDPStormDistActionAdvice()

	def deepCopy(self) -> "MDPStormActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPStormDistActionStrategy()

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPDictActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action according to a dictionary.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, d) -> None:
		self.d = d
		self.advice = MDPDictActionAdvice(d)

	def deepCopy(self) -> "MDPStormActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPDictActionStrategy(self.d)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPDTTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getDTSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, tree, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPDTActionStrategy(tree=tree)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runDTTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, tree, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getDTSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, tree,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPNNTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getNNSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPNNActionStrategy(model=model, threshold = threshold)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runNNTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getNNSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, model,threshold,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPMultiNNTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getMultiNNSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model1, threshold1:float, model2, threshold2:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPMultiNNActionStrategy(model1=model1, threshold1 = threshold1, model2=model2, threshold2 = threshold2)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runMultiNNTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model1, threshold1:float, model2, threshold2:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getMultiNNSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, model1,threshold1,model2,threshold2,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPStormTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getStormSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPStormActionStrategy(threshold = threshold)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runStormTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getStormSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, threshold,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPStormDistTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getStormDistSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPStormDistActionStrategy()
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runStormDistTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getStormDistSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPDictTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getDictSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, d:dict, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPDictActionStrategy(d)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runDictTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, d:dict, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getDictSimulationEngine(mdpState, mdpPredicates, mdpOperations, d, horizonTrace, quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPUniformTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getUniformSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPUniformActionStrategy()
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runUniformTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getUniformSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

def createPrismFile(layout,initAction=0):
	"""
	initAction
	-----------
	0 = all
	1 = east
	2 = west
	3 = north
	4 = south
	"""

	mdp, initState,initPredicates = readFromFile(layout)
	walls = mdp.walls
	holes = mdp.holes
	targets = mdp.targets
	position = initState.position

	height = len(walls)
	width = len(walls[0])
	file = createPrismFilefFromGrids(walls,holes,targets,position,initAction)
	return file

def getValueFromLayout(layout,formula_str = "Pmax=? [F win]"):
	prismFile = createPrismFile(layout, 0)
	value = getValue(prismFile,formula_str)
	# os.system("rm "+prismFile)
	return value

def getAllValuesFromLayout(layout):
	values = []
	for i in range(1,5):
		prismFile = createPrismFile(layout, i)
		value = getValue(prismFile)
		values.append(value)
		if DEBUG == True:
			print(prismFile)
		else:
			os.system("rm "+prismFile)
	return values

def getAllDistValuesFromLayout(layout):
	values = []
	for i in range(1,5):
		prismFile = createPrismFile(layout, i)
		formula1 = "Pmax=? [F win]"
		formula2 = "Tmin=? [F win]"
		value = getDistValue(prismFile,formula1,formula2)
		values.append(value)
		# if DEBUG == True:
		# 	print(prismFile)
		# else:
		# 	os.system("rm "+prismFile)
	return values
