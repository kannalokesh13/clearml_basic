from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml import Task
from clearml.automation.optuna import OptimizerOptuna


task = Task.init(project_name='example_classification2', task_name='experiment2', task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))



optimizer = HyperParameterOptimizer(

    base_task_id = "951bb4473c31471196807f3d327bc749",

     hyper_parameters=[
       UniformIntegerParameterRange('General/epochs', min_value=2, max_value=12, step_size=5),
       UniformParameterRange('General/learning_rate', min_value=0.001, max_value=0.01, step_size=0.002)
   ],

    objective_metric_title='Loss',
    objective_metric_series='train',
    objective_metric_sign='min',
    optimizer_class=OptimizerOptuna,
  
    # configuring optimization parameters
    execution_queue='default',  
    max_number_of_concurrent_tasks=2,  

    total_max_jobs=5,

    min_iteration_per_job=3,
    max_iteration_per_job=5


)


optimizer.start(job_complete_callback=job_complete_callback)

optimizer.set_time_limit(in_minutes=120.0)

optimizer.wait()

top_exp = optimizer.get_top_experiments(top_k=2)
print([t.id for t in top_exp])
# make sure background optimization stopped
optimizer.stop()

print('We are done, good bye')
