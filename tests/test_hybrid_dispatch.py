"""Dependency readiness must not dispatch unverified variants or duplicate jobs."""
from pkabench.runtime import require_compute
from pkabench.hybrid_dispatch import ready_tasks, taskkey

def test_readiness():
    tasks=[('x','AB','teacher'),('x','AB','catboost-17'),('x','A','teacher'),('x','A','catboost-17')]
    assert ready_tasks(tasks,set(),{})==[tasks[0],tasks[2]]
    done={taskkey(tasks[0])}
    assert ready_tasks(tasks,done,{taskkey(tasks[2]):'job'})==[tasks[1]]
    assert ready_tasks(tasks,done,{taskkey(tasks[1]):'job1',taskkey(tasks[2]):'job2'})==[]
    assert ready_tasks(tasks,{taskkey(t) for t in tasks},{})==[]

if __name__=='__main__':
    require_compute(); test_readiness(); print('hybrid rolling-dispatch readiness tests passed')
