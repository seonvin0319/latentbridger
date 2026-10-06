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
    return Agent.create(0, b['observations'], b['actions'], get_config())


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
    expected=agent.score(b['endpoint_targets'],b['endpoint_goals'],target=True)-agent.score(b['observations'],b['endpoint_goals'],target=True)
    np.testing.assert_allclose(delta,expected)


def test_ema_after_online_update(agent,batch):
    updated,info=agent.update(batch[0])
    assert all(np.isfinite(np.asarray(x)).all() for x in info.values())
    for name in ('phi','psi'):
        for old,new,target in zip(jax.tree_util.tree_leaves(agent.network.params['modules_target_'+name]),jax.tree_util.tree_leaves(updated.network.params['modules_'+name]),jax.tree_util.tree_leaves(updated.network.params['modules_target_'+name])):
            np.testing.assert_allclose(target,.005*new+.995*old,rtol=1e-5,atol=1e-7)


def test_rank_only_C_z_g_and_single_bypass():
    class Fake:
        def score(self,states,goals,**kwargs):
            # Fails if current observations or intermediate goals enter score.
            np.testing.assert_array_equal(goals,np.array([[9.,9.],[9.,9.]]))
            return states[:,0]
    candidates=jnp.array([[[2.,0.],[5.,0.]]])
    selected,best=Agent.rank_candidates(Fake(),candidates,jnp.array([[9.,9.]]))
    assert int(best[0]) == 1
    np.testing.assert_array_equal(selected,[[5.,0.]])
    class NoCritic:
        def score(self,*args,**kwargs): raise AssertionError('N=1 called critic')
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
