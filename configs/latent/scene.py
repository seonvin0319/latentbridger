from configs.latent._base import DEFAULT_VARIANT, latent_config


def get_config(variant=DEFAULT_VARIANT):
    return latent_config(
        env_name='scene-play-v0',
        horizon=25,
        discount=0.99,
        variant=variant,
    )
