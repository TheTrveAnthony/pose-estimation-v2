import sys as s

from functions import pose_estimate as pe



if len(s.argv) == 2:

	pe(s.argv[1])	

else:
	print("nope.")

