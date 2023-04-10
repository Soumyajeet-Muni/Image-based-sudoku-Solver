# Image-based-sudoku-Solver
This application helps you to extract Sudoku from an image and solve the sudoku. It  uses image processing, optical character recognition and backtracking

It uses a variety of image processing techniques to extract the Sudoku grid and optical character recognition to identify the digits from the image.Finally a backtracking algorithm is used to solve the Sudoku.
An ANN model is also trained to predict the digits from an image on the MNIST data-set. This model can be used to verify the accuracy of the digits predicted by optical character recognition.

The ANN is pre trained and saved.It can be used if the tesseract "image to string conversion" does not work properly

Getting Started:


1.Clone the Repository or download zip.

`git clone https://github.com/Soumyajeet-Muni/Image-based-sudoku-Solver.git` 
 

2.Move to the Current directory

`cd sudoku-solver`


Prerequisites:


1.Python 3.11+

2.Open CV

3.Tesseract-Ocr 

4.Pytesseract



How to Use:

python sudoku.py


Working:
1.Original Sudoku:



![image](https://user-images.githubusercontent.com/117106268/230973076-38bc0671-c24d-4e6b-aa1b-2bd64f6827b2.png)






2.After Preprocessing and Extraction:



![image](https://user-images.githubusercontent.com/117106268/230973173-160e9a50-5e7e-4d46-b901-ff4db2cf49fe.png)





3.Extraction of individual digits from the images



![image](https://user-images.githubusercontent.com/117106268/230973296-67705695-8422-4f00-b5ca-fffdcf34bb04.png)





4.Unsolved Sudoku Grid:



![image](https://user-images.githubusercontent.com/117106268/230973381-4a4164f5-7dcf-40f6-8fb7-02e496f859d2.png)





5.Solution:




![image](https://user-images.githubusercontent.com/117106268/230973558-9f92a7bf-1736-4c71-9eca-c18172378c1d.png)


