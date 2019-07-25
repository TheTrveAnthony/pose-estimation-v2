import torch as t 
import torchvision
#import numpy as np
from PIL import Image as im
from PIL import ImageDraw as idw 
#from matplotlib import pyplot as plt 	<=== For Some reason, importing matplolib makes the prediction return an error





def draw(img, p):
	
	"""This function will draw the predictions of the model
	img is our image, p is list containing a dictionnary containing all the predictions """
		
	########### First of all we gotta define which point connects together, there are points for the head(5), the shoulders, elbows, hands, top of thighs, knees, feet
	###########  top of thighs, knees and feet. We also gonna add 2 extra points for the neck and the pelvis, to make everythonh look better.
	########### We also provide the color of each line here
	connexions = [	
					(5, 7, 'navy'),		# shoulder => elbow
					(7, 9, 'navy'),		# elbow => hand
					(6, 8, 'navy'),		# same on the other side
					(8, 10, 'navy'),
					(11, 13, 'lime'),	# thigh => knee
					(13, 15, 'lime'),	# knee => foot
					(12, 14, 'lime'),	# same on the other side
					(14, 16, 'lime'),

					###### With The Extra points :

					(0, 17, 'aqua'),	# head => neck
					(17, 5, 'aqua'),	# neck => shoulders
					(17, 6, 'aqua'),
					(17, 18, 'teal'),	# neck => pelvis
					(18, 11, 'teal'),	# pelvis => thighs
					(18, 12, 'teal')
					]

	###### now let's find out how many objects were detected 
	
	l = len(p[0]["scores"])

	##### time to draw now, we'll only select objects with a score over .9

	d = idw.Draw(img)

	for k in range(l):

		if p[0]["scores"][k] > 0.98:

			##### Let's add the neck and pelvis:
			neck = (p[0]["keypoints"][k][5] + p[0]["keypoints"][k][6])/2
			pelv = (p[0]["keypoints"][k][11] + p[0]["keypoints"][k][12])/2

			#### it's getting tricky here

			nepe = t.zeros((2, 3))
			nepe[0] = neck ; nepe[1] = pelv 

			### Now let's put everything into a single tensor
			body = t.cat((p[0]["keypoints"][k], nepe))

			#### We can start drawing now, for real

			for tp in connexions:

				p0 = (int(body[tp[0], 0]), int(body[tp[0], 1]))
				p1 = (int(body[tp[1], 0]), int(body[tp[1], 1]))
				d.line([p0, p1], fill=tp[2], width=2)

			#### Now the points

			for ts in t.cat((body[0:1], body[5:])):
				d.ellipse((int(ts[0]-2), int(ts[1]-2), int(ts[0]+2), int(ts[1]+2)), 'fuchsia')

	### and finally
	#plt.imshow(np.asarray(img)) Not Like That
	img.show()


def pose_estimate(name):

	""" Makes the estimation """

	model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
	model.eval()	#### This is to set the model to evaluation mode 
	#### Open the image and convert it into a tensor
	image = im.open(name)
	image_tensor = torchvision.transforms.functional.to_tensor(image)

	### predict time
	
	output = model([image_tensor])			##### This returns a dictionnary #### For some reason, doing this in an external function returns an error
	

	### drawww
	draw(image, output)