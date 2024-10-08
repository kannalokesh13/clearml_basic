from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml import Task


Task.init(project_name='example_classification1', task_name='experiment2', task_type=Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(

    base_task_id = "e15e63d41e7f4ae1877975ec76b02c01",

     hyper_parameters=[
       UniformIntegerParameterRange('General/epochs', min_value=2, max_value=12, step_size=5),
       UniformParameterRange('General/learning_rate', min_value=0.0001, max_value=0.01, step_size=0.002)
   ],

    objective_metric_title='Loss',
    objective_metric_series='train',
    objective_metric_sign='min',


)


optimizer.start()

optimizer.wait()

optimizer.stop()
