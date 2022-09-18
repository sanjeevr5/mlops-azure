import os
import time
import numpy as np
import pandas as pd
import pickle
import joblib
import onnxruntime
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    global model, input_name, label_name, inputs_dc
    model_onnx = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'iris_dt.onnx')#Change to DT
    model = onnxruntime.InferenceSession(model_onnx, None)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    inputs_dc = ModelDataCollector(model_name="iris", designation="inputs", feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])

numpy_sample_input = NumpyParameterType(np.array([[5.1, 3.5, 1.4, 0.2], [1.1, 2.8, 4.4, 1.2]],dtype='float32'))
#standard_sample_input = StandardPythonParameterType(0.0)

sample_input = StandardPythonParameterType({'data': numpy_sample_input})

sample_output = NumpyParameterType(np.array([1.0, 1.0]))
outputs = StandardPythonParameterType({'results':sample_output}) 

@input_schema('Inputs', sample_input)

@output_schema(outputs)

def run(Inputs):
    global inputs_dc
    # the parameters here have to match those in decorator, both 'Inputs' and 
    # 'GlobalParameters' here are case sensitive
    try:
        data = Inputs['data']
        inputs_dc.collect(data)
        #assert isinstance(data, np.array)
        result = model.run([label_name], {input_name: data.astype(np.float32)})
        return result[0].tolist()
    except Exception as e:
        error = str(e)
        return error
