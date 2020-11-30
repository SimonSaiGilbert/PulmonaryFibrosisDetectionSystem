import tkinter as tk

def handle_sp():
	print("Handling select patient...")

def handle_ct():
	print("Handling view CT scan...")

def handle_seg():
	print("Handling segment lungs...")

def handle_fvc():
	print("Handling predict fvc...")

def main():
	window = tk.Tk()
	window.winfo_toplevel().title("F-PuPS")
	window.columnconfigure(0,weight=1,minsize=75)
	for i in range(5):
		window.rowconfigure(i,weight=1,minsize=50)

	frame_t = tk.Frame()
	label_t = tk.Label(text="Welcome to F-PuPS. Select the patient below and then choose your action.",master=frame_t)
	frame_t.grid(row=0,column=0,padx=5,pady=5)
	label_t.pack(padx=5,pady=5)

	frame_c = tk.Frame()
	label_c = tk.Label(text="Copyright 2020",master=frame_c)
	frame_c.grid(row=1,column=0,padx=5,pady=5)
	label_c.pack(padx=5,pady=5)

	frame_sp = tk.Frame()
	button_sp = tk.Button(text="Select patient...",master=frame_sp,command=handle_sp)
	frame_sp.grid(row=2,column=0,padx=5,pady=5)
	button_sp.pack(padx=5,pady=5)

	frame_actions = tk.Frame()

	button_ct = tk.Button(text="View CT Scan",master=frame_actions,command=handle_ct)
	button_ct.grid(row=0,column=0,padx=5,pady=5)

	button_seg = tk.Button(text="Segment Lungs",master=frame_actions,command=handle_seg)
	button_seg.grid(row=0,column=1,padx=5,pady=5)

	button_fvc = tk.Button(text="Predict FVC",master=frame_actions,command=handle_fvc)
	button_fvc.grid(row=0,column=2,padx=5,pady=5)

	frame_actions.grid(row=3,column=0,padx=5,pady=5)

	window.mainloop()

if __name__ == '__main__':
	main()
