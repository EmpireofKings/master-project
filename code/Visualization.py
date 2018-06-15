#!/usr/bin/python3.4
from tkinter import *
import time
from tkinter.messagebox import *
import random
import numpy as np
from time import time, sleep

#Output at the end of episode
#matrix1 = np.array([[2, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 8, 0, 1, 0]])
#matrix2 = np.array([[2, 2, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 8, 0, 1, 0]])
#matrix3 = np.array([[2, 2, 0, 1, 1],[1, 2, 1, 0, 1],[1, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 8, 0, 1, 0]])
#matrix4 = np.array([[2, 2, 0, 1, 1],[1, 2, 1, 0, 1],[1, 2, 0, 1, 1],[0, 0, 0, 0, 1],[1, 8, 0, 1, 0]])
#matrix5 = np.array([[2, 2, 0, 1, 1],[1, 2, 1, 0, 1],[1, 2, 0, 1, 1],[0, 2, 0, 0, 1],[1, 8, 0, 1, 0]])
#mat1 = matrix1.flatten()
#mat2 = matrix2.flatten()
#mat3 = matrix3.flatten()
#mat4 = matrix4.flatten()
#mat5 = matrix5.flatten()
#mat = np.array([mat1,mat2,mat3,mat4,mat5])

mat = [[2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,2,0,0,0,1,0,0,0,1,8,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,8,2,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0]]





#No. of iterations
i = 0

    
#Global class
class Game():
	
	#Class variables
    def __init__(self, master,matrix):
        self.matrix = matrix
        self.createButtons(master)

        self.bottomFrame = Frame(root)
        self.bottomFrame.grid(row=11, columnspan=20)

        self.quitBtn = Button(self.bottomFrame, text='Quit', command=self.quit)
        self.quitBtn.grid(row=13, columnspan=2)
	
	#Iterate over each cell, move to next column once row_max is reached
    def createButtons(self, parent):
        self.buttons = {}
        row = 0
        col = 0
        for x in range(0, 100):
            status = random.choice(['safe', 'danger'])
            self.buttons[x] = [
            Button(parent, bg='#8a8a8a'),
            status,
            row,
            col,
            [0 if status == 'danger' else 1]
            ]

            col += 1
            if col == 10:
                col = 0
                row += 1
            for k in self.buttons:
                self.buttons[k][0].grid(row= self.buttons[k][2], column= self.buttons[k][3])
				#Set the color to the different objects
                #Free space
                if self.matrix[k] == 0:
                    self.buttons[k][0].configure(bg = "white" )
                #Obstacle
                if self.matrix[k] == 1:
                    self.buttons[k][0].configure(bg = "black")
                #Path of drone
                if self.matrix[k] == 2:
                    self.buttons[k][0].configure(bg = "yellow")
                #Target
                if self.matrix[k] == 8:
                    self.buttons[k][0].configure(bg = "red")
            
    def quit(self):
        global root
        root.quit()


def main():
    global root
    root = Tk()
    root.title('Rescue Time!')
	
	#Iterating through the episodes
    def repeat():
        global i
        matrix = mat[i]
        i = i+1
        game = Game(root,matrix)
        root.after(600, repeat)
    
    i = 0
    while i < len(mat):
        repeat()
        root.mainloop()
    

if __name__ == '__main__':
    main()