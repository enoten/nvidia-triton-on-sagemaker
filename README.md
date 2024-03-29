# nvidia-triton-on-sagemaker
The original code and explation is presented <a href="https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/resnet50/triton_resnet50.ipynb">here</a>

The code is provided in the notebook <a href="https://github.com/enoten/nvidia-triton-on-sagemaker/blob/main/triron_endpoint_sagemaker_resnet50_deployment.ipynb">triron_endpoint_sagemaker_resnet50_deployment.ipynb
</a>

<h3>About this repo</h3>
This repo contains example of deploying ResNet50 (PyTorch *.pt file) model for Inference on AWS Sagemaker's Endpoint using Nvidia Triton Inference Server

<h3>Pt model</h3>
The way to create PyTorch model is explained in the <a href="https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/resnet50/triton_resnet50.ipynb">post</a>. 

To generate *.pt model please use <a href="https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/resnet50/workspace/pt_exporter.py">script</a> from <a href="https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-triton/resnet50/workspace"> this repo</a>
