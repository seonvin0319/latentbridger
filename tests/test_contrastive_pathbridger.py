"""Contract tests for contrastive replacement and preserved explicit policy."""
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from agents.contrastive_pathbridger import ContrastivePathBridgerAgent as Agent, get_config, infonce, progress_weights
from agents.pathbridger import PathBridgerAgent, FlowEndpointProposer, BridgeResidual, InverseDynamics
from utils.datasets import Dataset, PathBridgerDataset
from utils.goal_representation import goal_representation
from utils.flax_utils import save_agent, restore_agent
from utils.cpb_reference_bank import make_reference_goal_bank, checkpoint_reference_bank


@pytest.fixture(scope='module')
def batch():
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(120, 6)).astype(np.float32)
    terminals = np.zeros(120); terminals[[59, 119]] = 1
    dataset = PathBridgerDataset(Dataset.create(observations=observations, actions=rng.normal(size=(120, 2)).astype(np.float32), terminals=terminals), get_config())
    return dataset.sample(8), dataset


@pytest.fixture(scope='module')
def agent(batch):
    b, _ = batch
    return Agent.create(0, b['observations'], b['actions'], get_config(), reference_goal_bank=make_reference_goal_bank(batch[1],0))


def test_future_stays_in_episode(batch):
    _, dataset = batch
    for _ in range(50):
        starts = np.array([0, 30, 60, 90])
        b = dataset.sample(4, starts)
        indices = starts + b['value_offsets'].astype(int)
        assert np.all(indices > starts)
        assert np.all(indices <= np.array([59, 59, 119, 119]))
        np.testing.assert_array_equal(b['value_goals'], dataset.dataset['observations'][indices])


def test_infonce_diagonal():
    u = jnp.eye(8) * 3
    loss, info = infonce(u, u, repr_norm=True)
    wrong, _ = infonce(u, jnp.roll(u, 1, axis=0), repr_norm=True)
    assert float(loss) < float(wrong)
    assert float(info['critic/recall_at_1']) == 1
    expected = np.log(np.exp(9) + 7) - 9
    np.testing.assert_allclose(loss, expected, atol=1e-6)


@pytest.mark.parametrize('env,dim,indices', [('antmaze-medium-navigate-v0',29,[0,1]), ('cube-single-play-v0',28,[19,20,21]), ('cube-double-play-v0',37,[19,20,21,28,29,30]), ('puzzle-3x3-play-v0',55,list(range(20,55,4)))])
def test_goal_projection(env, dim, indices):
    x=jnp.arange(dim)[None]
    np.testing.assert_array_equal(goal_representation(x,'phi',env_name=env),np.asarray(indices)[None])


@pytest.mark.parametrize('step,lam', [(0,0),(99999,0),(100000,0),(150000,.5),(200000,1),(1000000,1)])
def test_schedule_and_stop_gradient(step, lam):
    d=jnp.array([-1., 0., 1., 20.])
    w, info=progress_weights(d,step)
    assert float(info['progress/lambda_C']) == lam
    assert np.max(w) <= 5
    if not lam: np.testing.assert_array_equal(w,np.ones(4))
    np.testing.assert_array_equal(jax.grad(lambda x: progress_weights(x,step)[0].sum())(d),np.zeros(4))
    raw=np.minimum(5,np.exp(np.clip(np.asarray(d)/(np.std(d)+1e-6),-2,2)))
    np.testing.assert_allclose(w,1-lam+lam*raw,rtol=1e-6)


def test_rank_only_unweighted():
    w,_=progress_weights(jnp.array([-100.,100.]),1000000,enabled=False)
    np.testing.assert_array_equal(w,np.ones(2))


def test_same_goal_delta(agent,batch):
    b,_=batch
    _,delta=agent._endpoint_weights(b['observations'],b['endpoint_goals'],b['endpoint_targets'])
    expected=agent.target_calibrated_score(b['endpoint_targets'],b['endpoint_goals'])-agent.target_calibrated_score(b['observations'],b['endpoint_goals'])
    np.testing.assert_allclose(delta,expected)


def test_ema_after_online_update(agent,batch):
    updated,info=agent.update(batch[0])
    assert all(np.isfinite(np.asarray(x)).all() for x in info.values())
    for name in ('phi','psi'):
        for old,new,target in zip(jax.tree_util.tree_leaves(agent.network.params['modules_target_'+name]),jax.tree_util.tree_leaves(updated.network.params['modules_'+name]),jax.tree_util.tree_leaves(updated.network.params['modules_target_'+name])):
            np.testing.assert_allclose(target,.005*new+.995*old,rtol=1e-5,atol=1e-7)


def test_rank_only_calibrated_z_g_and_single_bypass():
    class Fake:
        def calibrated_score(self,states,goals,**kwargs):
            # Fails if current observations or intermediate goals enter score.
            np.testing.assert_array_equal(goals,np.array([[9.,9.],[9.,9.]]))
            return states[:,0]
    candidates=jnp.array([[[2.,0.],[5.,0.]]])
    selected,best=Agent.rank_candidates(Fake(),candidates,jnp.array([[9.,9.]]))
    assert int(best[0]) == 1
    np.testing.assert_array_equal(selected,[[5.,0.]])
    class NoCritic:
        def calibrated_score(self,*args,**kwargs): raise AssertionError('N=1 called critic')
    selected,_=Agent.rank_candidates(NoCritic(),candidates[:,:1],jnp.array([[9.,9.]]))
    np.testing.assert_array_equal(selected,[[2.,0.]])


def test_preserved_policy_and_exact_pins(agent,batch):
    assert Agent._flow_endpoint_samples is PathBridgerAgent._flow_endpoint_samples
    assert Agent._sample_endpoint_candidates is PathBridgerAgent._sample_endpoint_candidates
    assert Agent.bridge_loss is PathBridgerAgent.bridge_loss
    assert Agent.idm_loss is PathBridgerAgent.idm_loss
    modules=agent.network.model_def.modules
    assert isinstance(modules['endpoint'],FlowEndpointProposer)
    assert isinstance(modules['bridge'],BridgeResidual)
    assert isinstance(modules['idm'],InverseDynamics)
    b,_=batch
    bridge=agent.construct_bridge(b['observations'],b['endpoint_targets'])
    np.testing.assert_array_equal(bridge[:,0],b['observations'])
    np.testing.assert_array_equal(bridge[:,-1],b['endpoint_targets'])
    assert agent._construct_bridge_prefix(b['observations'],b['endpoint_targets']).shape[1] == 6
    assert agent.sample_action_chunks(b['observations'],b['value_goals']).shape == (8,5,2)


def test_rewards_do_not_enter_update(agent,batch):
    b,_=batch
    a,_=agent.update(b)
    c,_=agent.update(dict(b,rewards=jnp.full((8,),jnp.nan),returns=jnp.full((8,),jnp.nan)))
    for x,y in zip(jax.tree_util.tree_leaves(a.network.params),jax.tree_util.tree_leaves(c.network.params)):
        np.testing.assert_array_equal(x,y)
    assert set(agent.network.model_def.modules) == {'phi','psi','target_phi','target_psi','endpoint','bridge','idm'}


def test_checkpoint(agent,batch,tmp_path):
    a,_=agent.update(batch[0])
    path=save_agent(a,tmp_path,1)
    b=restore_agent(agent,path)
    assert int(a.network.step)==int(b.network.step)
    for x,y in zip(jax.tree_util.tree_leaves(a.network.params),jax.tree_util.tree_leaves(b.network.params)):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('h',[1,2,5])
def test_execute_prefix_only(h):
    from types import SimpleNamespace
    from utils.contrastive_pathbridger_evaluation import evaluate
    class Policy:
        calls=0
        def sample_action_chunks(self,**kwargs):
            self.calls+=1
            return jnp.ones((1,5,1))
    class Env:
        spec=SimpleNamespace(max_episode_steps=10)
        action_space=SimpleNamespace(low=np.array([-1.]),high=np.array([1.]))
        def reset(self,**kwargs):
            self.steps=0
            return np.zeros(2),{'goal':np.ones(2)}
        def step(self,action):
            self.steps+=1
            return np.zeros(2),0.,self.steps==10,False,{'success':self.steps==10}
    policy=Policy()
    result=evaluate(policy,Env(),task_ids=[1],episodes_per_task=1,execute_h=h)
    assert policy.calls == int(np.ceil(10/h))
    assert result['success_count']==1


def test_regularizer_and_ties():
    u=jnp.zeros((8,4))
    loss,info=infonce(u,u)
    np.testing.assert_allclose(loss,np.log(8)+.01*np.log(8)**2,rtol=1e-6)
    assert float(info['critic/recall_at_1']) == 1/8
    assert float(info['critic/positive_rank']) == 4.5


def test_endpoint_weight_cannot_train_critic(agent,batch):
    full=agent.replace(network=agent.network.replace(step=300000))
    grads=jax.grad(lambda params: full.endpoint_loss(batch[0],params,full.rng)[0])(full.network.params)
    for name in ('phi','psi','target_phi','target_psi'):
        assert all(np.count_nonzero(np.asarray(x))==0 for x in jax.tree_util.tree_leaves(grads['modules_'+name]))


def test_original_policy_numerically_preserved(agent,batch):
    from agents.pathbridger import get_config as original_config
    b,_=batch
    original=PathBridgerAgent.create(0,b['observations'],b['actions'],original_config())
    for name in ('endpoint','bridge','idm'):
        for x,y in zip(jax.tree_util.tree_leaves(agent.network.params['modules_'+name]),jax.tree_util.tree_leaves(original.network.params['modules_'+name])):
            np.testing.assert_array_equal(x,y)
    for method in ('bridge_loss','idm_loss'):
        actual=getattr(agent,method)(b,agent.network.params)[0]
        expected=getattr(original,method)(b,original.network.params)[0]
        np.testing.assert_array_equal(actual,expected)
    _,info=original.update(b)
    assert all(np.isfinite(np.asarray(x)) for x in info.values())


def test_geometric_gamma(batch):
    _, dataset = batch
    np.random.seed(192)
    starts = np.zeros(40000, dtype=np.int64)
    goals, _ = dataset._sample_goal_indices(starts, np.full_like(starts, 59), dataset.critic_p)
    for k in (2, 10, 30, 59):
        assert abs(np.mean(goals >= k) - dataset.discount ** (k - 1)) < .012


def test_reference_future_marginal_and_rng(batch, monkeypatch):
    _, dataset = batch
    calls=[]
    original=dataset.sample
    def spy(size):
        b=original(size); calls.append(b['value_goals'].copy()); return b
    monkeypatch.setattr(dataset,'sample',spy)
    np.random.seed(12)
    state=np.random.get_state()
    bank=make_reference_goal_bank(dataset,3)
    actual=np.random.random(4)
    np.random.set_state(state)
    np.testing.assert_array_equal(actual,np.random.random(4))
    assert bank.shape==(512,2) and len(calls)==1
    np.testing.assert_array_equal(bank,np.asarray(goal_representation(calls[0],'phi',env_name='antmaze-medium-navigate-v0')))
    np.testing.assert_array_equal(bank,make_reference_goal_bank(dataset,3))
    assert not np.array_equal(bank,make_reference_goal_bank(dataset,4))


def test_log_partition_stability_and_anchor_offset_invariance():
    from agents.contrastive_pathbridger import log_mean_exp
    reference=jnp.array([[10000.,10001.,9999.],[-10000.,-9999.,-10001.]])
    raw=jnp.array([10002.,-9998.])
    offset=jnp.array([500.,-3000.])
    expected=raw-log_mean_exp(reference)
    assert np.isfinite(np.asarray(expected)).all()
    np.testing.assert_allclose(raw+offset-log_mean_exp(reference+offset[:,None]),expected,atol=.002)
    ordinary=jnp.array([[1.,2.,3.]])
    np.testing.assert_allclose(log_mean_exp(ordinary),np.log(np.exp(np.asarray(ordinary)).mean(axis=1)),rtol=1e-6)


def test_calibrated_api_and_cache(agent,batch):
    b,_=batch
    raw=agent.raw_score(b['observations'],b['value_goals'])
    partition=agent.log_partition(b['observations'],agent.reference_goal_bank)
    calibrated=agent.calibrated_score(b['observations'],b['value_goals'])
    np.testing.assert_allclose(calibrated,raw-partition,atol=1e-6)
    assert not np.allclose(raw,calibrated)
    cached=agent.with_reference_cache()
    np.testing.assert_allclose(cached.calibrated_score(b['observations'],b['value_goals']),calibrated,rtol=1e-5,atol=1e-5)
    updated,_=cached.update(b)
    assert not updated.reference_cache_valid
    fresh=updated.with_reference_cache()
    np.testing.assert_allclose(updated.calibrated_score(b['observations'],b['value_goals']),fresh.calibrated_score(b['observations'],b['value_goals']),rtol=1e-5,atol=1e-5)


def test_exact_resume_including_bank_optimizer_and_sampler(agent,batch,tmp_path):
    _,dataset=batch
    np.random.seed(329)
    a,_=agent.update(dataset.sample(8))
    path=save_agent(a,tmp_path,1)
    next_batch=dataset.sample(8)
    expected,_=a.update(next_batch)
    restored=restore_agent(agent.replace(reference_goal_bank=jnp.zeros_like(agent.reference_goal_bank)),path)
    np.testing.assert_array_equal(checkpoint_reference_bank(path),agent.reference_goal_bank)
    np.testing.assert_array_equal(restored.reference_goal_bank,agent.reference_goal_bank)
    resumed_batch=dataset.sample(8)
    for key in next_batch:
        np.testing.assert_array_equal(next_batch[key],resumed_batch[key])
    actual,_=restored.update(resumed_batch)
    for x,y in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
        np.testing.assert_array_equal(x,y)


def test_calibrated_progress_same_goal_and_not_raw():
    class Fake:
        config={'progress_scale':1.,'variant':'cpb_full'}
        network=type('Network',(),{'step':300000})()
        def target_calibrated_score(self,states,goals):
            np.testing.assert_array_equal(goals,[[7.,8.],[7.,8.]])
            return states[:,0]-3*states[:,1]
        def raw_score(self,*args,**kwargs):
            raise AssertionError('Progress must not use raw score')
    s=jnp.array([[0.,0.],[1.,1.]])
    z=jnp.array([[2.,1.],[4.,0.]])
    _,delta,_=Agent._progress(Fake(),s,jnp.array([[7.,8.],[7.,8.]]),z)
    np.testing.assert_array_equal(delta,[-1.,6.])


def test_evaluation_cache_does_not_change_training_graph(agent,batch):
    plain,_=agent.update(batch[0])
    cached,_=agent.with_reference_cache().update(batch[0])
    for x,y in zip(jax.tree_util.tree_leaves(plain.network),jax.tree_util.tree_leaves(cached.network)):
        np.testing.assert_array_equal(x,y)
