import os.path
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
from PIL import ImageTk,Image 
from segmentation.preprocess_data import load_data
from segmentation.gui_segment_pipeline import segmentation_fn
import matplotlib.pyplot as plt

class Fpups(ttk.Frame):
	def __init__(self,window):
		ttk.Frame.__init__(self, window)
		window.columnconfigure(0,weight=1,minsize=500)
		for i in range(5):
			window.rowconfigure(i,weight=1,minsize=50)

		self.filename = None
		self.patient_data = None
		self.patient_slices = []

		self.frame_t = tk.Frame()
		self.label_t = tk.Label(text="Welcome to F-PuPS. Select the patient below and then choose your action.",master=self.frame_t)
		self.frame_t.grid(row=0,column=0,padx=5,pady=5)
		self.label_t.pack(padx=5,pady=5)

		self.frame_c = tk.Frame()
		self.label_c = tk.Label(text="Copyright 2020",master=self.frame_c)
		self.frame_c.grid(row=1,column=0,padx=5,pady=5)
		self.label_c.pack(padx=5,pady=5)

		self.frame_sp = tk.Frame()
		self.button_sp = tk.Button(text="Select patient...",master=self.frame_sp,command=self.handle_sp)
		self.frame_sp.grid(row=2,column=0,padx=5,pady=5)
		self.button_sp.pack(padx=5,pady=5)

		self.frame_actions = tk.Frame()

		self.button_ct = tk.Button(text="View CT Scan",master=self.frame_actions,command=self.handle_ct)
		self.button_ct.grid(row=0,column=0,padx=5,pady=5)

		self.button_seg = tk.Button(text="Segment Lungs",master=self.frame_actions,command=self.handle_seg)
		self.button_seg.grid(row=0,column=1,padx=5,pady=5)

		self.button_fvc = tk.Button(text="Predict FVC",master=self.frame_actions,command=self.handle_fvc)
		self.button_fvc.grid(row=0,column=2,padx=5,pady=5)

		self.frame_actions.grid(row=3,column=0,padx=5,pady=5)

	def load_patient_slices(self):
		#loads patient data and makes list of each slice
		self.patient_data = load_data(self.filename)
		for i in range(self.patient_data.shape[0]):
			self.patient_slices.append(self.patient_data[i][:][:])

	def close_window():
		popup.destroy()

	def handle_sp(self):
		print("Handling select patient...")
		# self.filename = '/Users/sgilbert/Desktop/ec601-term-project-gui-dev/ID00007637202177411956430'
		self.filename = askdirectory()
		try:
			assert(os.path.isdir(self.filename))
			print("Ready to process patient from: " + str(self.filename))
			self.label_c.config(text = "Ready to process: " + str(self.filename))
			self.button_sp.config(text = "Select another patient...")
		except:
			pass

	def handle_ct(self):
		print("Handling view CT scan...")
		try:
			assert(os.path.isdir(self.filename))
			print("Displaying CT scan data for: " + str(self.filename))

			self.load_patient_slices()

			window = tk.Toplevel(root)
			window.wm_title("CT Scan")

			l = tk.Label(window, text="Enter desired slice in range 1 - %d"%self.patient_data.shape[0])
			l.grid(row=0, column=0)

			global slice_val_entry
			slice_val_entry = ttk.Entry(window)
			slice_val_entry.grid(row=0, column=1)
			button_calc = tk.Button(window, text="Display Slice", command=self.get_slice_and_display)
			button_calc.grid(row=1, column=0)

			root.mainloop() 

		except:
			popup = tk.Toplevel()
			popup.winfo_toplevel().title("Warning")
			popup.geometry("250x50")
			label = tk.Label(text="Please select a patient first.",master=popup)
			label.pack()
			button = tk.Button(text="Close",master=popup,command=popup.destroy)
			button.pack()

	def get_slice_and_display(self):
		try:
			slice_val = int(slice_val_entry.get()) - 1
			#print("slice val: ", slice_val)
			root2 = tk.Toplevel(root)
			root2.wm_title("Slice %d"%(slice_val+1))
			canvas = tk.Canvas(root2, width=self.patient_data.shape[1], height=self.patient_data.shape[2])  
			img = ImageTk.PhotoImage(image=Image.fromarray(self.patient_data[slice_val][:][:]))  
			canvas.create_image(0, 0, anchor=tk.NW, image=img) 
			canvas.pack()  
			root.mainloop() 
		except:
			print("The slice number you entered is not valid")

		



	def display_image(self, slice_val):
		root2 = tk.Toplevel(root) 
		canvas = tk.Canvas(root2, width=self.patient_data.shape[1], height=self.patient_data.shape[2])  
		img = ImageTk.PhotoImage(image=Image.fromarray(self.patient_data[slice_val][:][:]))  
		canvas.create_image(20, 20, anchor=tk.NW, image=img) 
		canvas.pack()  
		root.mainloop() 


	def get_segmented_slice_and_display(self):
		try:
			slice_val = int(segmented_slice_val_entry.get()) - 1
			#print("slice val: ", slice_val)
			root2 = tk.Toplevel(root)
			root2.wm_title("Segmented Slice %d"%(slice_val+1))
			canvas = tk.Canvas(root2, width=self.segmentation_data.shape[1], height=self.segmentation_data.shape[2])  
			img = ImageTk.PhotoImage(image=Image.fromarray(256*self.segmentation_data[slice_val][:][:]))  
			canvas.create_image(0, 0, anchor=tk.NW, image=img) 
			canvas.pack()  
			root.mainloop() 
		except:
			print("The slice number you entered is not valid")

	def handle_seg(self):
		print("Handling segment lungs...")
		try:
			assert(os.path.isdir(self.filename))
			print("Displaying segmented lung data for: " + str(self.filename))
			# insert code for segmenting lungs and displaying results here
			self.segmentation_data = segmentation_fn(self.filename)
            
			self.load_patient_slices()

			window = tk.Toplevel(root)
			window.wm_title("CT Scan")

			l = tk.Label(window, text="Enter desired slice in range 1 - %d"%self.patient_data.shape[0])
			l.grid(row=0, column=0)

			global segmented_slice_val_entry
			segmented_slice_val_entry = ttk.Entry(window)
			segmented_slice_val_entry.grid(row=0, column=1)
			button_calc = tk.Button(window, text="Display Segmented Slice", command=self.get_segmented_slice_and_display)
			button_calc.grid(row=1, column=0)

			root.mainloop() 
            
		except:
		 	popup = tk.Toplevel()
		 	popup.winfo_toplevel().title("Warning")
		 	popup.geometry("250x50")
		 	label = tk.Label(text="Please select a patient first.",master=popup)
		 	label.pack()
		 	button = tk.Button(text="Close",master=popup,command=popup.destroy)
		 	button.pack()

	def handle_fvc(self):
		print("Handling predict fvc...")
		try:
			assert(os.path.isdir(self.filename))
			print("Predicting FVC for: " + str(self.filename))
			# insert code for predicting FVC here
		except:
			popup = tk.Toplevel()
			popup.winfo_toplevel().title("Warning")
			popup.geometry("250x50")
			label = tk.Label(text="Please select a patient first.",master=popup)
			label.pack()
			button = tk.Button(text="Close",master=popup,command=popup.destroy)
			button.pack()

def main():
	global root
	root = tk.Tk()
	root.winfo_toplevel().title("F-PuPS")
	fpups = Fpups(root)
	root.mainloop()

if __name__ == '__main__':
	main()
