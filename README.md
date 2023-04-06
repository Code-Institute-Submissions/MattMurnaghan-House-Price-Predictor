# **House-Price-Predictor**
The aim of this repo is to demonstrate Machine Learning and Deep Learning concepts with respect to predicting the price of houses for sale given a dataset supplied by kaggle.com

## **Project Set Up**
I approached this project a bit differently than the other projects I've undertaken as part of the Full Stack Developer course with the Code Institute. 

Due to a recent issue with the GitPod platform that I used to develop my previous projects, I took it upon myself to configure my projects locally. I'm working on a Macbook Pro with an M1 Pro chip, so I have more than enough processing power to run the Machine Learning and Deep Learning processing.

To configure the project locally, I checked the supported Python versions by Heroku. As some of the libraries used in this project are not supported on the latest version of Python, 3.11.2 currently, I will be deploying the project on Python 3.10.10, which is available on all stacks on Heroku.

In order to maintain the project environment, I used the pyenv version management project to create a virtual environment on my local machine to run the project in. This ensures that the project will remain stable when it is reproduced by other developers in the future.

## **About Pyenv**

Pyenv is a tool for managing multiple versions of Python on a single machine. It allows you to easily switch between different versions of Python, and provides a way to manage Python dependencies for different projects.

### **Why Use Pyenv?**

Using Pyenv has several benefits:

- **Compatibility:** Different projects may require different Python versions. Pyenv allows you to switch between versions of Python without affecting other projects running on the same machine.

- **Dependency Management:** Pyenv provides a way to manage Python dependencies for different projects. With Pyenv, you can create isolated virtual environments for each project and install the required dependencies in these environments.

- **Flexibility:** Pyenv is a flexible tool that can be customized to fit your workflow. You can specify a default Python version to use across all projects, set per-project Python versions, and specify Python versions on a per-shell basis.

### **Installation**

To install Pyenv, follow the instructions in the [official documentation](https://github.com/pyenv/pyenv#installation). Once installed, you can start using Pyenv to manage your Python environment.

## **Getting Started**

To get started with Pyenv, check out the [official documentation](https://github.com/pyenv/pyenv#usage). This will guide you through the process of creating virtual environments, switching between Python versions, and managing dependencies for different projects.

## **Setting up the Virtual Environment (venv)**
You can check for available Python versions that you can install with pyenv by using the following command:
```
pyenv install --list

OR

pyenv install -l
```

I installed the necessary Python version required for the project using this command in the terminal:
```
pyenv install 3.10.10
```

After cloning the repository, I used the command below to set the local python environment for the directory that housed the project:
```
pyenv local 3.10.10
```

This creates a .python-version file in the current directory. Please note, you must first navigate to the project directory in the terminal before running this command. You can do this by typing this command in the terminal:
```
cd /path/to/your/project/your_project_name
```

You can check that this has worked by checking for hidden files in the directory.
Run this command:
```
ls -a
```

You can check the version is correct with this command:
```
cat .python-version
```

Once the local version of python was set, I used the following Python command to create the virtual environment:
```
python -m venv .venv
```

This creates a virtual environment in the directory, containing all of the binaries necessary to run Python. It also has it's own pip, which allows you t keep all of the packages and dependencies needed for the package stored locally in the venv, instead of on your global system. *Clean!*

The common naming convention is to use .venv, but you can use any name you like.

I included the .venv file and the .python-version file in my .gitignore file, so you will need to configure this yourself when you are replicating the project. It is good practice to do this, as virtual environments should be configured wherever they are being deployed for program stability.

After creating your environment, you should activate it by running this command from the root directory of the project:
```
. .venv/bin/activate
```

The images below show how the terminal looks before and after you have successfully activated your virtual environment.

![Terminal - global environment](images/global_terminal.jpg)
*Terminal before venv has been activated*

<br>

![Terminal - venv environment](images/venv_terminal.jpg)
*Terminal after venv has been activated*

<br>

You can check to make sure that the correct version of python is being detected in the environment by running the following command:
```
python -V
```

It should output the following message to the terminal if you have done everything right:
```
Python 3.10.10
```

To deactivate the environment, simply type the following command from anywhere int the terminal:
```
deactivate
```

This should bring you back into the global terminal.

Now the the virtual environment in set up, we can start intalling the dependencies needed for this project.

## **Installing packages and dependencies**
to install the packages and required dependencie for this project, run the following command from the root directory:

```
pip install requiremets.txt
```

Because I am running on Mac OS using the ARM64 architecure from the M1 Pro processor, there are some slight changes I had to make when installing the packages.

I ran the following command to ensure that only the binaries using x86 architecture were installed, as not all of the python packages used in this project have been released with compatible versions for my architecture.

```
env ARCHFLAGS="-arch x86_64" pip install -r requirements.txt
```

This ensured that the project was stable and ran locally on my machine. I also replaced tensorlfow with tensorflow-macos in the requirements.txt file as per apples direction in this article: [Using TensorFlow on MacOS](https://developer.apple.com/forums/thread/686926)

Once the required packages have been installed, you can see the full list by running the following command:
```
pip list
```

After installing the package on my system, I executed the following command to ensure that the project's package versions are compatible with its stable version. This ensures that anyone replicating the project will also use compatible package versions.
```
pip freeze > requirements.txt
```

Now that the environment is configured, we can begin to explore the data set and  the relating business case for this project.