import win32gui 
import win32ui 
from ctypes import windll
from PIL import Image, ImageGrab, ImageChops, ImageOps, ImageEnhance
import numpy as np
from numpy import array
from skimage.measure import compare_ssim
import imutils
import pyautogui
import time
import math
import struct 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

CLIENT = 'NoxPlayer'
WINDOW_SUBSTRING = 'Nox'

MAX_POINT = 5 

def process_is_running(process_name): #If the process is running then return its pid
	for proc in psutil.process_iter():
		try:
			if process_name.lower() == proc.name().lower():
				print(proc.pid)
				return proc.pid
				#print(proc.name())

		except(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass
	return 0
	
#Gets coordinate of the program 
def get_window_info(): 
	#set window info
	window_info = {}
	win32gui.EnumWindows(set_window_coordinate, window_info)
	return window_info 

#EnumWindows handlers 
def set_window_coordinate(hwnd, window_info):
	if win32gui.IsWindowVisible(hwnd):
		if WINDOW_SUBSTRING in win32gui.GetWindowText(hwnd):
			rect = win32gui.GetWindowRect(hwnd)
			x = rect[0]
			y = rect[1]
			w = rect[2] - x 
			h = rect[3] - y 
			window_info['x'] = x
			window_info['y'] = y 
			window_info['w'] = w 
			window_info['h'] = h 
			window_info['name'] = win32gui.GetWindowText(hwnd) 
			win32gui.SetForegroundWindow(hwnd) 	

#Grabs program's img. 
def get_screen(x1, y1, x2, y2):
	box = (x1, y1, x2, y2)
	screen = ImageGrab.grab(box) 

	img = array(screen.getdata(), dtype=np.uint8).reshape((screen.size[1], screen.size[0], 3))

	return img

# Group rectangles
def grp_rectangle(recs, epso):
	# Group rectangles
	for i in range(len(recs)):
		recs.append(recs[i])

	grp_rect, weights = cv2.groupRectangles(recs, 1, epso)

	# for i in grp_rect:
	# 	(x, y, w, h) = i[0], i[1], i[2], i[3]
	# 	cv2.rectangle(cropped_im3, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# 	cv2.imshow('Title 2', cropped_im3)

	return grp_rect


# Using SSIM [Structure Similarity Index] to find the differences
def method_SSIM(image1, image2):
	#Convert images to numpy array 
	np_im1 = np.array(image1)
	np_im2 = np.array(image2) 

	#Convert images to reality color 
	np_im1 = cv2.cvtColor(np_im1, cv2.COLOR_BGR2RGB)
	np_im2 = cv2.cvtColor(np_im2, cv2.COLOR_BGR2RGB)

	#Convert images to grayscale 
	im1_g = cv2.cvtColor(np_im1, cv2.COLOR_BGR2GRAY)
	im2_g = cv2.cvtColor(np_im2, cv2.COLOR_BGR2GRAY) 

	#Blur the images in order to reduce noises 
	im1_b = cv2.GaussianBlur(im1_g, (5, 5), 0)
	im2_b = cv2.GaussianBlur(im2_g, (5, 5), 0)

	#Using SSIM formular to find the differences 
	(score, diff) = compare_ssim(im1_b, im2_b, full=True)
	diff = (diff * 255).astype("uint8")

	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	rects = []
	for c in cnts: 
			# compute the bounding box of the contour and then draw the
			# bounding box on both input images to represent where the two
			# images differ
		if cv2.contourArea(c) >= 10:
			(x, y, w, h) = cv2.boundingRect(c)
			rects.append((x, y, w, h))
			cv2.rectangle(np_im1, (x, y), (x + w, y + h), (0, 255, 255), 2) # Draw with yellow 
			

	if len(rects) > MAX_POINT:
		rects = grp_rectangle(rects, 1)

		for i in rects:
			(x, y, w, h) = i[0], i[1], i[2], i[3]
			cv2.rectangle(np_im1, (x, y), (x + w, y + h), (255, 0, 255), 2) # Draw with purple 
	
	cv2.imshow('Title1', np_im1)

	return rects 

# Using Image_Chop to find the differences
def medthod_image_chop(image1, image2):
	# Enhance color of the images
	# cropped_im1 = ImageEnhance.Color(cropped_im1)
	# cropped_im1 = cropped_im1.enhance(2.0)

	# cropped_im2 = ImageEnhance.Color(cropped_im2)
	# cropped_im2 = cropped_im2.enhance(1.0)

	# Gets the differences between 2 images. 
	cropped_im3 = ImageChops.difference(image1, image2)

	cropped_im3 = ImageEnhance.Color(cropped_im3)
	cropped_im3 = cropped_im3.enhance(5.0)

	# Convert PIL to OpenCV
	cropped_im3 = np.array(cropped_im3)


	# Convert to binary images to apply contours 
	gray = cv2.cvtColor(cropped_im3, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0) # Reduce noises in the images.
	ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

	# Draw contours
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	rects = []
	# Loops over the contours and store all the coordinate of differences into a dictornary
	for c in cnts:
		# Compute the bounding box of the contour and thhen draw the
		# bounding box on both input images to represent where the 2 images differ
		if cv2.contourArea(c) >= 10:
			(x, y, w, h) = cv2.boundingRect(c)
			rects.append((x, y, w, h))
			#rectsUsed.append(False) # Set all bool is False
			#dict_rects[(x, y, w, h)] = w * h
			cv2.rectangle(cropped_im3, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draw with REd

	if len(rects) > MAX_POINT: 
		# Group rectangles
		epso = 0.1
		while len(rects) != MAX_POINT:
			rects = grp_rectangle(rects, epso).tolist()
			epso += 0.1
		# for i in range(len(recs)):
		# 	rects.append(recs[i])

		# grp_rect, weights = cv2.groupRectangles(rects, 1, 1)
		# rects = grp_rect
		for i in rects:
			(x, y, w, h) = i[0], i[1], i[2], i[3]
			cv2.rectangle(cropped_im3, (x, y), (x + w, y + h), (0, 255, 0), 2) # DRaw with Green
			

	cv2.imshow('Title 2', cropped_im3)
	return rects

# Auto Click
def click_points(win_info, coor_rects):
	for i in coor_rects:
		(x, y, w, h) = i[0], i[1], i[2], i[3]
		pyautogui.click(win_info['x'] + (x + w/2), win_info['y'] + 138 + (y + h/2))
		time.sleep(2)


user32 = windll.user32
user32.SetProcessDPIAware()

# Detect Nox's location on Windows 
win_info = [] 
win_info = get_window_info() 
winimg = get_screen(win_info['x'], win_info['y'], win_info['w'] +
					 win_info['x'], win_info['h'] + win_info['y'])

# Change to real color
screen_im = Image.fromarray(winimg, 'RGB')

#Crops the image 
cropped_im1 = screen_im.crop((0, 138, win_info['w'], 544))
cropped_im2 = screen_im.crop((0, 547, win_info['w'], 953))

recs = []
recs = method_SSIM(cropped_im1, cropped_im2)


if len(recs) != MAX_POINT: 
	recs = medthod_image_chop(cropped_im1, cropped_im2)
	
print(len(recs))
click_points(win_info, recs)
cv2.waitKey(0)


