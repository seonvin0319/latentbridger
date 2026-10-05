from configs.latent_endpoint_chunk._base import DEFAULT_VARIANT, endpoint_chunk_config


def get_config(variant=DEFAULT_VARIANT):
    return endpoint_chunk_config(
        env_name='cube-double-play-v0', discount=0.99, variant=variant
    )
