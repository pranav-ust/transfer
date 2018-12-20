inp = open("text.txt", "r+")
oup = open("output_base.txt", "r+")

precision = 0
for x, y in zip(inp, oup):
	a_x = set(x.rstrip().split(":")[1].split(","))
	a_y = set(y.rstrip().split(":")[1].split(","))
	#precision += len(a_x.intersection(a_y))
	precision += len(a_x.intersection(a_y)) / len(a_x)
print("Average Precision is ", (precision / 857) )

inp.close()
oup.close()
