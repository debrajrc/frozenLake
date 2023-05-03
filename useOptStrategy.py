import stormpy, glob
from conditionalMinDist import *
from frozenLakeMdpClasses import createPrismFile

# def investigateModel(model): # this is for debugging
# 	for state in model.states:
# 		actionOffset = 0
# 		for action in state.actions:
# 			# print("action",action.id,action.transitions)
# 			for transition in action.transitions:
# 				actionIndex = model.get_choice_index(state,actionOffset)
# 				# print(actionIndex)
# 				# actionLabel = model.choice_labeling.get_labels_of_choice(1)
# 				print("From state {}, with action {}, with probability {}, go to state {}".format(state,action.id, transition.value(),transition.column))
# 			actionOffset += 1
# 	# print(type(model.labeling))
# 	for state in model.states:
# 		print("State", state, "has label:", model.labeling.get_labels_of_state(state))

def getOptDist(prismFile, formula_str1, formula_str2):
	# formula_str1 = "Pmax=? [F win]"
	program = stormpy.parse_prism_program(prismFile)
	formulas = stormpy.parse_properties_for_prism_program(formula_str1, program)
	model = stormpy.build_model(program, formulas)
	# print(model)
	result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
	assert result.has_scheduler
	scheduler = result.scheduler
	print(type(scheduler))
	# print(scheduler)
	assert scheduler.memoryless
	assert scheduler.deterministic
	dtmc = model.apply_scheduler(scheduler)
	assert dtmc.model_type == stormpy.ModelType.MDP
	for state in dtmc.states:
			assert len(state.actions) == 1
	formulas = stormpy.parse_properties_for_prism_program(formula_str2, program)
	result = stormpy.model_checking(dtmc, formulas[0], extract_scheduler=True)
	# assert result.has_scheduler
	initial_state = dtmc.initial_states[0]
	value = result.at(initial_state)
	return value



def main():
	layouts = glob.glob("layout/*.lay")
	# layouts = glob.glob("test.lay")
	f = open("results.csv","w")
	print("layout, probability, opt cond exp dist, cond exp dist storm",file=f)
	for layout in layouts:
		# print(layout)
		prismFile = createPrismFile(layout)
		formula_str1 = "Pmax=? [F win]"
		formula_str2 = 'Tmin=? [F win]'
		# formula_str3 = "Tmax=? [F win]"
		v = getDistValue(prismFile,formula_str1,formula_str2)
		# v2 = getDistValue(prismFile,formula_str1,formula_str2)
		# print("Storm",v1)
		# print("our code",v2)
		# v3 = getDistValue(prismFile,formula_str1,formula_str3)
		print(f"{layout}, {v[0]}, {v[1]}, {v[2]}",file=f)
		print(f"{getOptDist(prismFile, formula_str1, formula_str2)},{layout}, {v[0]}, {v[1]}, {v[2]}")
		# print(f"{layout}, {v1[0]}, {v2[0]}, {v1[1]}, {v2[1]}, {v3[1]}")

if __name__ == "__main__":
	main()