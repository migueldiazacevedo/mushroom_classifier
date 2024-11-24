# Mushroom Classification -- Computer Vision with PyTorch 
![shroom](https://mycologyst.art/images/midjourney/mycologist/mycology-book-0004-2176x544.jpg)

According to [Brandenburg and Ward](https://pubmed.ncbi.nlm.nih.gov/30062915/) (Mycologia, 2018), in the years between 1999 and 2016 there were approximately 7428 cases of mushroom exposure. Mushroom poisoning is remarkably dangerous and enabling foragers to detect the particular species of a mushroom is a powerful tool for ensuring the safety of those who go out into nature and anyone with interests in fungi more broadly.

The aim of the current project is to utilize machine learning frameworks in **PyTorch** in order to implement a **neural network classifier** capable of **computer vision** in order to define the species of mushroom from tensors of pixel data.

Moreover, this project uses **transfer learning** in order to leverage ResNet-18 to create more sophisticated and accurate neural nets than those made from scratch.

With these methods, we are capable of generating a performant model capable of mushroom prediction. 

_see __mushroom_vision.ipynb__ for performance metrics_

### Data
[Mushroom classification - Common genus's images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)

### Project Contents
- __mushroom_vision.ipynb__:<br> the main notebook. See how the data was processed, how the PyTorch Model was built, and model performance.  
- __utils__:<br> a python package containing utility functions
- __requirements.txt__: <br>all dependencies
- __.gitignore__
- __LICENSE__


### Usage
I recommend creating a virtual environment, in this case I call it "mushroom_vision".

In terminal:
```terminal
python -m venv mushroom_vision 
```
Activate venv in terminal
```
source mushroom_vision/bin/activate
```
side note: can deactivate venv with 
```terminal
deactivate
```
Install all requirements by first going to the directory where requirements.txt is (e.g. project root directory) 
```terminal
cd name/of/root/directory
```
and then typing in terminal:
```terminal
pip install -r requirements.txt
```

Now you are ready to run the Jupyter notebooks mushroom_vision.ipynb using your favorite IDE or 
```terminal
jupyter lab
```
Step through the notebook sequentially to gain an understanding of my workflow and the neural network that I generated.

### Requirements
See full list of requirements with exact versions to recreate my development environment in requirements.txt<br><br>

__Key Requirements__:
- kaggle
- jupyter
- matplotlib
- numpy
- pandas
- pillow
- scikit-learn
- seaborn 
- torch
- torchinfo
- torchvision

### License
[MIT](https://opensource.org/license/mit)  

#### Contact
Miguel A. Diaz-Acevedo at migueldiazacevedo@gmail.com