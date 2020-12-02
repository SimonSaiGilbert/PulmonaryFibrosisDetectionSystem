import os.path
import tkinter as tk
from tkinter.filedialog import askdirectory

class Fpups:
	def __init__(self,window):
		window.columnconfigure(0,weight=1,minsize=500)
		for i in range(5):
			window.rowconfigure(i,weight=1,minsize=50)

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

	def close_window():
		popup.destroy()

	def handle_sp(self):
		print("Handling select patient...")
		global filename 
		filename = askdirectory()
		try:
			assert(os.path.isdir(filename))
			print("Ready to process patient from: " + str(filename))
			self.label_c.config(text = "Ready to process: " + str(filename))
			self.button_sp.config(text = "Select another patient...")
		except:
			pass

	def handle_ct(self):
		print("Handling view CT scan...")
		try:
			assert(os.path.isdir(filename))
			print("Displaying CT scan data for: " + str(filename))
			# insert code for displaying CT data here
		except:
			popup = tk.Toplevel()
			popup.winfo_toplevel().title("Warning")
			popup.geometry("250x50")
			label = tk.Label(text="Please select a patient first.",master=popup)
			label.pack()
			button = tk.Button(text="Close",master=popup,command=popup.destroy)
			button.pack()

	def handle_seg(self):
		print("Handling segment lungs...")
		try:
			assert(os.path.isdir(filename))
			print("Displaying segmented lung data for: " + str(filename))
			# insert code for segmenting lungs and displaying results here
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
			assert(os.path.isdir(filename))
			print("Predicting FVC for: " + str(filename))
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
	root = tk.Tk()
	root.winfo_toplevel().title("F-PuPS")
	fpups = Fpups(root)
	root.mainloop()

if __name__ == '__main__':
	main()