"""Resource and priority gates for the persistent experiment queue."""
from scripts.run_goalspace_ablation import available_gpus, plan, HARD, completed
from scripts import run_goalspace_ablation as queue


def test_busy_gpu_is_not_available_even_with_plenty_of_memory():
    gpus=[dict(index=i,uuid=str(i),free_mib=32000) for i in range(4)]
    assert available_gpus(gpus, {'0','1'}, {'2'}) == [gpus[3]]
    assert available_gpus(gpus[:1], {'0'}, set()) == []
    assert available_gpus([dict(index=0,uuid='0',free_mib=1000)],set(),set()) == []


def test_priority_and_config_unavailable_are_respected():
    jobs=plan([dict(env=e,status='available' if e!='scene' else 'config unavailable') for e in HARD])
    assert [(j['env'],j['variant']) for j in jobs[:2]] == [('puzzle_3x3','gs_trl_weighted'),('cube_double','gs_trl_weighted')]
    assert all(j['seed']==0 for j in jobs[:11])
    assert all(j['seed'] in (1,2) and j['variant']=='gsdtrl_weighted' for j in jobs[11:])
    assert not any(j['env']=='scene' for j in jobs)
    assert len(jobs)==19


def test_smoke_or_incomplete_evaluations_cannot_skip_full_run(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(queue,'OUT',tmp_path)
    job=dict(env='puzzle_3x3',variant='gs_trl_weighted',seed=0)
    run=queue.run_dir(job);run.mkdir(parents=True)
    path=run/'complete.json'
    path.write_text(json.dumps(dict(steps=1,smoke=True)))
    assert not completed(job)
    path.write_text(json.dumps(dict(steps=1000000,smoke=False)))
    assert not completed(job)
    for step in (100000,300000,500000,800000,1000000):
        for h in (5,2,1):(run/f'evaluation_{step}_h{h}.json').write_text('{}')
    assert not completed(job)
    for step in (100000,300000,500000,800000,1000000):
        for h in (5,2,1):
            (run/f'evaluation_{step}_h{h}.json').write_text(json.dumps(dict(checkpoint=step,h=h,num_tasks=5,episodes_per_task=50,seed=0,variant='gs_trl_weighted')))
    assert completed(job)


def test_opt_in_sharing_keeps_local_reservation_and_memory_gate():
    gpu=dict(index=0,uuid='0',free_mib=31000)
    assert available_gpus([gpu],{'0'},set(),allow_shared=True)==[gpu]
    assert available_gpus([gpu],{'0'},{'0'},allow_shared=True)==[]
    assert available_gpus([dict(gpu,free_mib=1000)],{'0'},set(),allow_shared=True)==[]
