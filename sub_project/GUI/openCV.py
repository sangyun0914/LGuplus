# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np

# Define function to show frame
def show_frames():
  # Get the latest frame and convert into Image
  src= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
  src = cv2.resize(src,None,fx=0.5,fy=0.5)

  # status
  img_status = np.zeros((int(height/2),WIDTH_INFOR,3) , dtype=np.uint8)
  img_status = Image.fromarray(img_status)

  imgtk_status = ImageTk.PhotoImage(image = img_status)
  status.imgtk = imgtk_status
  status.configure(image = imgtk_status)

  # webcam
  img_cam = Image.fromarray(src)

  imgtk_cam = ImageTk.PhotoImage(image = img_cam)
  webcam.imgtk = imgtk_cam
  webcam.configure(image=imgtk_cam)

  # feedback
  img_feedback = np.zeros((int(height/2),WIDTH_INFOR,3) , dtype=np.uint8)
  img_feedback = Image.fromarray(img_feedback)

  imgtk_feedback = ImageTk.PhotoImage(image = img_feedback)
  feedback.imgtk = imgtk_feedback
  feedback.configure(image = imgtk_feedback)

  # Repeat after an interval to capture continiously
  webcam.after(20, show_frames)

if __name__ == "__main__":
  # Create an instance of TKinter Window or frame
  win = Tk()

  cap= cv2.VideoCapture(0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  WIDTH_INFOR = 180
  GEOMETRY = str(int(width/2)+2*int(WIDTH_INFOR))+"x"+str(int(height/2))

  # Set the size of the window
  win.geometry(GEOMETRY)

  # status label
  status = Label(win)
  status.grid(row=0, column=0)

  # Create a Label to capture the Video frames
  webcam = Label(win)
  webcam.grid(row=0, column=1)

  # feedback label
  feedback = Label(win)
  feedback.grid(row=0,column=2)
  show_frames()
  win.mainloop()