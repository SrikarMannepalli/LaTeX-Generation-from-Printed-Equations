# LaTeX Generation from Printed Equations


Project for the course _Digital Image Processing_, Monsoon - 2019, IIIT-Hyderabad.  
Team members : Srikar Mannepalli(20171138) , Venkata Sathya  Vivek(20171182)
	
Our project is implementation of this paper -  https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Brewer_Sun.pdf.
This can also be found in the below mentioned in References directory.
## Project Description
The main goal is to build a system that takes a scan, PDF screenshot or a photograph of a printed mathematical equation and produces a LaTeX code representation that can generate the mathematical equation.

LaTeX is a powerful typesetting system that is extremely useful for technical documents, particularly mathematical equations. Working with lengthy mathematical equations can be a tedious and error-prone process. But, once rendered, the output cannot be modified, as we don’t have access to the underlying code. The ability to take a screenshot or a photograph of an existing equation and generate the LaTeX code for it can be extremely useful.

## Package Requirements

Following python3 packages have been used in this project - 
* opencv
* scikit-image
* matplotlib
* numpy
The project has been done using Python3.7. Kindly use Python 3.6+ only.
Command to install these packages has been provided in the below section. 

## Usage

Clone this repository - [https://github.com/SrikarMannepalli/LaTeX-Generation-from-Printed-Equations.git](https://github.com/SrikarMannepalli/LaTeX-Generation-from-Printed-Equations.git)
>git clone https://github.com/SrikarMannepalli/LaTeX-Generation-from-Printed-Equations.git
>cd LaTeX-Generation-from-Printed-Equations && cd src

Download dataset from the following Google Drive link - 
https://drive.google.com/file/d/1ZVzMHzEOer2FmO1hRBNswRetothb-jVQ/view?usp=sharing

The folder has some images in Clean and Images folders which can be used to run the code.

The following command installs the required packages. Please ensure that pip3 corresponds to python3.6+. Else replace command accordingly. req.txt has all the packages mentioned above that are needed to run the code. 
>pip3 -r req.txt

Run main.py with the image's filepath as argument
>python main.py <path_to_image>

Ensure that the python command in the above line corresponds to python3.6+.

## Output
	
On running the code, different images show up corresponding to different steps in the pipeline. At the end, the latex code string corresponding to  equation is printed on the command window. The last image shown corresponds to the image formed from the latex code generated. 
Note- Used matplotlib for this purpose. In case of limits being present, matplotlib shows 'limit' next to the integration symbol while it does not occur when using any other online tool to convert. That 'limit' is because of the inner details of matplotlib and is not a wrong matching.

## Testing

The above process has been tested on two linux machines.
