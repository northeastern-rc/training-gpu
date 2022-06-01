# June2021-GPUs
Material for June 2021 GPU bootcamp module

## What's in this training?
* Training PowerPoint slides "GPUs-on-Discovery.pptx"
* Exercises - practical examples on how to access and utilize GPUs on Discovery
  * Exercise_C - CUDA in C for Saxpy and how to submit it on Discovery using sbatch
  * Exercise_HelloWorld - Helloworld CUDA example in C
  * tf - Jupyter notebooks to compare CPU vs GPU performance for matrix multiplication using Tensorflow
  * pytorch - Jupyter notebook for Pytorch example of matrix multiplication
  * cuda - CUDA example of vector addition, expected output, memory check for cuda, and related files 
  * Supplementary Material - Extra slides and examples on GPU and CUDA
  
## Steps to download and use the repo on Discovery
1. Login to a Discovery shell or use the [Discovery OnDemand interface](https://rc-docs.northeastern.edu/en/latest/first_steps/connect_ood.html).

2. Enter your desired directory within Discovery and download the training material. For example:
```bash
cd $HOME
git clone git@github.com:<repo-location/repo-name>.git
cd <repo-name>
```
3. Download the training slides to your local computer, where you have access to PowerPoint to open the slides. Follow the slides to execute different examples.
