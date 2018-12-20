import json

filename = open("training", "r+")
aids = open("aids.txt", "w+")

user = ""
aid = set()

for line in filename:
	temp_dict = json.loads(line.rstrip())
	if ("user_gp_frequency" in temp_dict):
		if (temp_dict["user_id"] != user): #If we get a new user
			if ("article_l1_categories" in temp_dict):
				aids.write(",".join(i for i in aid) + "\n")
				user = temp_dict["user_id"]
				print("New user\n")
				print("User is ", user)
				aids.write(user + ":")
				aid = set()
				aid.update([temp_dict["article_contentid"]])
		else: #If the user is same
			if ("article_l1_categories" in temp_dict):
				aid.update([temp_dict["article_contentid"]])

print(user)
print(aid)

filename.close()
aids.close()
