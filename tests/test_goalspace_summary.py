"""Only matched, independently recorded seeds support the final statistic."""
import csv
import json
from scripts import summarize_goalspace_ablation as summary


def test_three_seeds_require_matched_configs_and_use_sample_std(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'OUT',tmp_path)
    def save(seed,score,power=0.5):
        run=tmp_path/'puzzle_3x3/gsdtrl_weighted'/f'seed{seed}'
        run.mkdir(parents=True,exist_ok=True)
        (run/'config.json').write_text(json.dumps({'agent':{'value_distance_weight_power':power}}))
        (run/'evaluation_1000000_h5.json').write_text(json.dumps(dict(
            checkpoint=1000000,h=5,num_tasks=5,episodes_per_task=50,
            overall_success=score,N=32,temperature=1.0)))
    def result():
        summary.main()
        with (tmp_path/'gsdtrl_multiseed.csv').open() as stream:
            return next(r for r in csv.DictReader(stream) if r['env']=='puzzle_3x3' and r['h']=='5')
    save(0,0.1);save(1,0.2)
    assert result()['mean']==''
    save(2,0.3)
    row=result()
    assert float(row['mean'])==20
    assert float(row['sample_std'])==10
    save(2,0.3,power=1.0)
    row=result()
    assert row['mean']=='' and row['status']=='config verification required'
