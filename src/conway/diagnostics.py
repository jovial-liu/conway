"""A non-desktop multimodal protocol probe. It never executes a returned action."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from .actions import validate_action
from .brain import OpenAICompatibleVLM, OpenCUABrain
from .computer import MockComputer
from .config import ConwayConfig
from .hardware import detect_hardware
from .runtime import connect_external, start_local_runtime
import os


@contextmanager
def model_session(config: ConwayConfig, state_root: Path, *, tool_manifest=None):
    """Share model selection and cleanup across non-desktop diagnostics."""
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    if config.api_key_env and not api_key:
        raise ValueError(f'Configured API-key environment variable {config.api_key_env} is empty')
    runtime, brain = None, None
    try:
        runtime = (connect_external(config.endpoint, config.model, api_key=api_key) if config.endpoint
                   else start_local_runtime(detect_hardware(), state_root, config.profile, config.port,
                        runtime_path=config.runtime_path, startup_timeout=config.startup_timeout, context_size=config.context_size))
        model = runtime.served_model or runtime.model_repo
        use_opencua = config.brain == 'opencua' or (config.brain == 'auto' and ('opencua' in model.lower() or config.profile == 'computer-use'))
        options = {'api_key': api_key, 'timeout': config.request_timeout}
        brain = (OpenCUABrain(runtime.base_url, model, min_pixels=config.opencua_min_pixels,
                             max_pixels=config.opencua_max_pixels, **options) if use_opencua
                 else OpenAICompatibleVLM(runtime.base_url, model, tool_manifest=tool_manifest, **options))
        yield brain, {'model': model, 'brain': 'opencua' if use_opencua else 'generic',
                      'profile': runtime.profile_name if hasattr(runtime, 'profile_name') else config.profile}
    finally:
        try:
            if brain:
                brain.close()
        finally:
            if runtime:
                runtime.close()


def probe_model(config: ConwayConfig, state_root: Path) -> dict:
    with model_session(config, state_root) as (brain, metadata):
        with TemporaryDirectory(prefix='conway-probe-') as directory:
            observation = MockComputer(Path(directory)).observe(1)
            result = brain.decide('This is a connection test on a synthetic image. Choose wait. No real task or desktop is present.',
                                  'Synthetic observation; no previous actions.', observation)
            action = validate_action(result.action, observation.width, observation.height)
        return {'status': 'protocol_ok', **metadata,
                'image_request_accepted': True, 'valid_action_type': action['type'],
                'actions_executed': 0, 'vision_grounding_verified': False,
                'note': 'The endpoint accepted an image and returned a valid action. This is not a GUI-task or visual-accuracy benchmark.'}
