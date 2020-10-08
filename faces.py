import numpy as np
import cv2
import pickle
import time
import os
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
import random

ALL_TIME = 0
start_time = 0
end_time = 0
workers={}
every_10_minutes_count = 0
conf = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")
labels = {}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

for value in labels.values():
	workers[value] = 0

cap = cv2.VideoCapture(0)

while(True):


	ret, frame = cap.read()

	start_time = time.perf_counter()

	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
	for (x, y, w, h) in faces:

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]


		id_, conf = recognizer.predict(roi_gray)

		# if conf>=45:

		#print(labels[id_])
		font = cv2.FONT_HERSHEY_SIMPLEX
		name = labels[id_]
		color = (255, 255, 255)
		stroke = 2
		cv2.putText(frame, "PavelPetrov", (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#print(conf)

		color = (255, 0, 0)
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('frame',frame)

	end_time = time.perf_counter()
	ALL_TIME = ALL_TIME + (end_time - start_time)
	if conf>=45:
		workers[name] += end_time - start_time
	if ALL_TIME>=2:
		ALL_TIME = 0
		every_10_minutes_count+=1
		for worker, worked_time in workers.items():

			if worked_time>1:
				worked_time = 2 - random.uniform(0,0.06)
			#print(worked_time)
			with open('day_results/{}.txt'.format(worker), 'a') as f:
				f.write('{} \n '.format(worked_time))
				workers[worker] = 0
	if cv2.waitKey(20) & 0xFF == ord("q"):
		break

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
day_results_dir = os.path.join(BASE_DIR, "day_results")
for root, dirs, files in os.walk(day_results_dir):
	for file in files:
		with open('day_results/{}'.format(file), 'r+') as f:
			content = f.read()
			content_10mins = content.split('\n')
			content_10mins = content_10mins[:len(content_10mins)-1]
			#print(content_10mins)
			proc = list(map(lambda x:(float(x)/2)*100, content_10mins))
			round_proc_y = [round(x,2) for x in proc]
			minutes_x = [2*x for x in range(1,len(round_proc_y)+1)]
			plt.xlabel('секунды')
			plt.ylabel('процент присутствия на работе')
			name = file.split('.')[0]
			plt.plot(minutes_x,round_proc_y)
			plt.savefig('{}.png'.format(name))

cap.release()
cv2.destroyAllWindows()
