#!/usr/bin/env python
###################################################################################
#bookreader.py
#Takes a picture of a page of printed text, performs Optical Character Recognition (ORC) 
#on the image, and then reads it aloud.
#
#Karan Nayan
#31 Oct,2013
#Dexter Industries
#www.dexterindustries.com/BrickPi
#
#You may use this code as you wish, provided you give credit where it's due.
###################################################################################
import time
from subprocess import call
import re
import cv2


#Function splits a big paragraph into smaller sentences for easy TTS
def splitParagraphIntoSentences(paragraph):
    sentenceEnders = re.compile('[.!?]')
    sentenceList = sentenceEnders.split(paragraph)
    return sentenceList

#Calls the Espeak TTS Engine to read aloud a sentence
#	-ven+m7:	Male voice
#	-s180:		set reading to 180 Words per minute
#	-k20:		Emphasis on Capital letters
def sound(spk):
        cmd_beg=" espeak -ven+m7 -s180 -k20 --stdout '"
        cmd_end="' | aplay"
        print cmd_beg+spk+cmd_end
        call ([cmd_beg+spk+cmd_end], shell=True)


while True:
	#Take an image from the RaspberryPi camera with sharpness 100(increases the readability of the text for OCR)
	call ("raspistill -o j2.jpg -t 1 -sh 100", shell=True)
	print "Image taken"

        image = cv2.imread("j2.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        cv2.imwrite("j3.jpg", thresh)

	
	#Start the Tesseract OCR and save the text to out1.txt
	call ("tesseract j3.jpg out1", shell=True)
	print "OCR complete"
        
        	
	#Open the text file and split the paragraph to Sentences
	fname="out1.txt"
	f=open(fname)
	content=f.read()
	print content
	sentences = splitParagraphIntoSentences(content)

	#Speak aloud each sentence in the paragraph one by one
	for s in sentences:
		sound(s.strip())
