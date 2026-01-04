## A Volume is created called as "mlruns" which will be shared across projects.

## To run the Docker Image, following is the command - 

* docker build -t model-serving .
* docker run -d --name model-serving -p 8085:8085 -v mlruns2:/mlruns2 model-serving
