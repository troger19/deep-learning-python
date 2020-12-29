When saving the model, it must have a special structure. It must start with version number, therefore its model.save("saved_model/1")

pull docker image

`docker pull tensorflow/serving`

map your model in structure saved_model/1  saved_model/2 ..
map config file models.config in which there is path to model in docker container. (not in my computer, in docker!!!)

`docker run -p 8501:8501 -v "D:\Java\deep-learning-python\Classification\2\saved_model:/models/model" -v "D:\Java\deep-learning-python\Classification\2\saved_model:/models/model" tensorflow/serving --model_config_file=/models/model/models.config`

get all models

`http://localhost:8501/v1/models/mymodel`

###prediction
POST 
INPUT is picture in 3-D array in JSON

`http://localhost:8501/v1/models/mymodel:predict`

Multiple versions REST API

