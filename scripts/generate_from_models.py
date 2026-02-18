#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class ConfigError(Exception):
    pass


STANDARD_MODEL_KEYS = {"name", "modelname", "provider", "parameters", "env", "command"}


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def env_to_ordered_dict(env: Any) -> Dict[str, str]:
    if env is None:
        return {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    if isinstance(env, list):
        parsed: Dict[str, str] = {}
        for item in env:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                parsed[key] = value
            elif isinstance(item, str):
                parsed[item] = ""
            else:
                raise ConfigError(f"Unsupported env item type: {type(item)}")
        return parsed
    raise ConfigError(f"Unsupported env type: {type(env)}")


def merge_env(base_env: Any, override_env: Any) -> Dict[str, str]:
    merged = env_to_ordered_dict(base_env)
    merged.update(env_to_ordered_dict(override_env))
    return merged


def build_provider_index(providers: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    provider_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    defaults_per_type: Dict[str, int] = {}

    for provider in providers:
        p_type = provider.get("type")
        p_subtype = provider.get("subtype")
        if not p_type or not p_subtype:
            raise ConfigError("Each provider must define both 'type' and 'subtype'.")

        key = (str(p_type), str(p_subtype))
        if key in provider_index:
            raise ConfigError(f"Duplicate provider type/subtype detected: {key}")

        provider_index[key] = provider
        defaults_per_type[key[0]] = defaults_per_type.get(key[0], 0) + (1 if key[1] == "default" else 0)

    for p_type, count in defaults_per_type.items():
        if count > 1:
            raise ConfigError(f"Provider type '{p_type}' has more than one default subtype.")

    return provider_index


def resolve_provider_parameters(
    provider_index: Dict[Tuple[str, str], Dict[str, Any]],
    provider_type: str,
    provider_subtype: str,
    memo: Dict[Tuple[str, str], Dict[str, Any]],
    stack: List[Tuple[str, str]],
) -> Dict[str, Any]:
    key = (provider_type, provider_subtype)

    if key in memo:
        return copy.deepcopy(memo[key])

    if key in stack:
        chain = " -> ".join([f"{t}/{s}" for t, s in stack + [key]])
        raise ConfigError(f"Provider inheritance cycle detected: {chain}")

    provider = provider_index.get(key)
    if provider is None:
        raise ConfigError(f"Provider not found: type={provider_type}, subtype={provider_subtype}")

    stack.append(key)
    resolved: Dict[str, Any] = {}
    parent_subtype = provider.get("inheritsfrom")
    if parent_subtype:
        parent_key = (provider_type, str(parent_subtype))
        if parent_key not in provider_index:
            raise ConfigError(f"Missing parent provider for {provider_type}/{provider_subtype}: {provider_type}/{parent_subtype}")
        resolved = resolve_provider_parameters(provider_index, provider_type, str(parent_subtype), memo, stack)

    resolved = deep_merge(resolved, provider.get("parameters", {}))
    stack.pop()

    memo[key] = copy.deepcopy(resolved)
    return copy.deepcopy(resolved)


def resolve_models(models_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    models = models_cfg.get("models")
    providers = models_cfg.get("providers")

    if not isinstance(models, list):
        raise ConfigError("models.yml must contain top-level list 'models'.")
    if not isinstance(providers, list):
        raise ConfigError("models.yml must contain top-level list 'providers'.")

    provider_index = build_provider_index(providers)
    provider_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    resolved_models: List[Dict[str, Any]] = []

    for model in models:
        if not isinstance(model, dict):
            raise ConfigError("Each model entry must be an object/mapping.")

        name = model.get("name")
        if not name:
            raise ConfigError("Each model requires a 'name'.")

        provider_ref = model.get("provider")
        if not isinstance(provider_ref, dict):
            raise ConfigError(f"Model '{name}' missing provider object.")

        p_type = provider_ref.get("type")
        p_subtype = provider_ref.get("subtype")
        if not p_type or not p_subtype:
            raise ConfigError(f"Model '{name}' provider must include type and subtype.")

        params = resolve_provider_parameters(provider_index, str(p_type), str(p_subtype), provider_cache, stack=[])
        if "parameters" in model:
            params = deep_merge(params, model["parameters"])

        params["env"] = merge_env(params.get("env"), model.get("env"))
        if "command" in model:
            params["command"] = copy.deepcopy(model["command"])

        resolved = {
            "name": name,
            "modelname": model.get("modelname", name),
            "provider_type": str(p_type),
            "provider_subtype": str(p_subtype),
            "parameters": params,
        }

        for key, value in model.items():
            if key not in STANDARD_MODEL_KEYS:
                resolved[key] = copy.deepcopy(value)

        resolved_models.append(resolved)

    return resolved_models


def lines_from_text(text: str) -> List[str]:
    if not text:
        return []
    return text.splitlines()


def format_env_list(env: Dict[str, str], indent: int = 6) -> List[str]:
    prefix = " " * indent
    return [f"{prefix}- {key}={value}" for key, value in env.items()]


def format_command_block(command: Any, indent: int = 4) -> List[str]:
    prefix = " " * indent
    if isinstance(command, str):
        lines = command.rstrip("\n").splitlines()
        block = [f"{prefix}command: >"]
        block.extend([f"{prefix}  {line}" for line in lines])
        return block
    if isinstance(command, list):
        if len(command) >= 5:
            first = ", ".join(json.dumps(item) for item in command[:3])
            rest = ",".join(json.dumps(item) for item in command[3:])
            rendered = f"[{first},{rest}]"
        else:
            rendered = "[" + ", ".join(json.dumps(item) for item in command) + "]"
        return [f"{prefix}command: {rendered}"]
    raise ConfigError(f"Unsupported command type: {type(command)}")


def append_yaml_block(output_lines: List[str], key: str, value: Any, indent: int = 4) -> None:
    prefix = " " * indent
    if key == "restart" and value is False:
        output_lines.append(f"{prefix}{key}: no")
        return
    if isinstance(value, (str, int, float)):
        output_lines.append(f"{prefix}{key}: {value}")
        return
    if isinstance(value, bool):
        output_lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        return

    output_lines.append(f"{prefix}{key}:")
    block = yaml.safe_dump(value, sort_keys=False, default_flow_style=False).rstrip("\n")
    for line in block.splitlines():
        output_lines.append(f"{prefix}  {line}")


def append_build_and_image(output_lines: List[str], build_value: Any, indent: int = 4) -> None:
    if isinstance(build_value, dict) and "image" in build_value:
        build_dict = copy.deepcopy(build_value)
        image_value = build_dict.pop("image")
        append_yaml_block(output_lines, "build", build_dict, indent=indent)
        append_yaml_block(output_lines, "image", image_value, indent=indent)
        return
    append_yaml_block(output_lines, "build", build_value, indent=indent)


def generate_compose_text(config: Dict[str, Any], resolved_list: List[Dict[str, Any]], resolved_models: Dict[str, Dict[str, Any]]) -> str:
    compose_cfg = config["compose"]
    output_lines: List[str] = []
    output_lines.extend(lines_from_text(compose_cfg.get("header", "")))

    order = [model["name"] for model in resolved_list]

    container_name_overrides = compose_cfg.get("container_name_overrides", {})
    env_remove_keys = set(compose_cfg.get("env_remove_keys", []))
    env_add_by_model = compose_cfg.get("env_add_by_model", {})
    env_value_overrides = compose_cfg.get("env_value_overrides", {})

    for index, model_name in enumerate(order):
        model = resolved_models.get(model_name)
        if model is None:
            raise ConfigError(f"Unknown model in resolved model list: {model_name}")

        params = model["parameters"]
        env = copy.deepcopy(params.get("env", {}))
        for key in list(env.keys()):
            if key in env_remove_keys:
                env.pop(key, None)
        if model_name in env_add_by_model:
            env.update(env_add_by_model[model_name])
        for key, value in list(env.items()):
            if key in env_value_overrides:
                env[key] = str(env_value_overrides[key])
        command = params.get("command")
        if command is None:
            raise ConfigError(f"Model '{model_name}' has no resolved command.")

        if index > 0:
            output_lines.append("")

        output_lines.append(f"  {model_name}:")

        container_name = container_name_overrides.get(model_name, model_name)
        output_lines.append(f"    container_name: {container_name}")

        output_lines.append("    environment:")
        output_lines.extend(format_env_list(env, indent=6))

        if "build" in params:
            append_build_and_image(output_lines, params["build"], indent=4)
        if "image" in params and not isinstance(params["image"], dict):
            append_yaml_block(output_lines, "image", params["image"], indent=4)

        image_cfg = params.get("image", {})
        if isinstance(image_cfg, dict):
            if "build" in image_cfg:
                append_build_and_image(output_lines, image_cfg["build"], indent=4)
            if "image" in image_cfg:
                append_yaml_block(output_lines, "image", image_cfg["image"], indent=4)
        elif image_cfg:
            append_yaml_block(output_lines, "image", image_cfg, indent=4)

        for key in ["volumes", "restart", "networks", "deploy", "ipc", "ulimits"]:
            if key in params:
                append_yaml_block(output_lines, key, params[key], indent=4)

        healthcheck_cfg = compose_cfg.get("verbose_healthcheck")
        if healthcheck_cfg:
            append_yaml_block(output_lines, "healthcheck", healthcheck_cfg, indent=4)

        output_lines.extend(format_command_block(command, indent=4))

    footer = compose_cfg.get("footer", "")
    if footer:
        output_lines.append("")
        output_lines.append("")
        output_lines.extend(lines_from_text(footer))

    return "\n".join(output_lines) + "\n"


def generate_swap_text(config: Dict[str, Any], resolved_list: List[Dict[str, Any]], resolved_models: Dict[str, Dict[str, Any]]) -> str:
    swap_cfg = config["swap"]
    output_lines: List[str] = []
    output_lines.extend(lines_from_text(swap_cfg.get("header", "")))

    order = [model["name"] for model in resolved_list]

    macro_up = swap_cfg.get("cmd_up_macro", "model_up")
    macro_down = swap_cfg.get("cmd_down_macro", "model_down")

    for index, model_name in enumerate(order):
        model = resolved_models.get(model_name)
        if model is None:
            raise ConfigError(f"Unknown model in resolved model list: {model_name}")

        params = model["parameters"]
        ttl = model.get("ttl", params.get("ttl", 600))
        concurrency = model.get("concurrencyLimit", params.get("concurrencyLimit", 4096))

        if index > 0:
            output_lines.append("")

        output_lines.append(f"  {model_name}:")
        output_lines.append(f"    proxy: http://{model_name}:8000")
        output_lines.append(f"    cmd: ${{{macro_up}}} {model_name}")
        output_lines.append(f"    cmdStop: ${{{macro_down}}} {model_name}")
        output_lines.append(f"    ttl: {ttl}")
        output_lines.append(f"    concurrencyLimit: {concurrency}")

    footer = swap_cfg.get("footer", "")
    if footer:
        output_lines.append("")
        output_lines.extend(lines_from_text(footer))

    return "\n".join(output_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docker-compose and llama-swap configs from models.yml")
    parser.add_argument("config", type=Path, help="Path to generator config YAML")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError("Generator config root must be a mapping/object.")

    models_path = Path(config.get("models_file", ""))
    if not models_path:
        raise ConfigError("Generator config must define models_file.")
    if not models_path.is_absolute():
        models_path = (args.config.parent / models_path).resolve()

    with models_path.open("r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)

    resolved_list = resolve_models(models_cfg)
    resolved_models = {m["name"]: m for m in resolved_list}

    compose_text = generate_compose_text(config, resolved_list, resolved_models)
    swap_text = generate_swap_text(config, resolved_list, resolved_models)

    compose_out = Path(config["compose"]["output_file"])
    swap_out = Path(config["swap"]["output_file"])
    if not compose_out.is_absolute():
        compose_out = (args.config.parent / compose_out).resolve()
    if not swap_out.is_absolute():
        swap_out = (args.config.parent / swap_out).resolve()

    compose_out.parent.mkdir(parents=True, exist_ok=True)
    swap_out.parent.mkdir(parents=True, exist_ok=True)

    compose_out.write_text(compose_text, encoding="utf-8")
    swap_out.write_text(swap_text, encoding="utf-8")

    print(f"Wrote {compose_out}")
    print(f"Wrote {swap_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)