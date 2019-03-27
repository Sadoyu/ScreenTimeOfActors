from tkinter import *
 
from tkinter import ttk

from tkinter import filedialog
import test
#functions:

def getFilePath():
 
    window.filename = filedialog.askopenfilename()
    print(window.filename)
    text.delete(0, END)
    text.insert(0, window.filename)
    global name
    name = window.filename

def Main():
    filepath = name
    print(filepath)
    test.main(filepath)
    

#create window
window = Tk()
 
window.title("Screen Time of Actors")

window.geometry('700x1000')

#project name
label = Label(window, text='SCREEN TIME OF ACTORS IN A MOVIE', fg="white", bg="red")

label.config(font=("Courier", 18, 'bold'))

label.place(x=150 , y=5 )

#text box
text = Entry(window,width=57)
 
text.place(x=150, y=45)


#get file path button
btn = Button(window, text="Insert", command=getFilePath)
 
btn.place(x=500, y=45)


#Main button
btn = Button(window, text="START", command=Main) 
btn.place(x=350, y=70)

window.mainloop()
