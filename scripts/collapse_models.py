#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class ConfigError(Exception):
    pass


class LiteralStringDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralStringDumper.add_representer(str, _str_presenter)


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


def env_to_dict(env: Any) -> Dict[str, Any]:
    if env is None:
        return {}
    if isinstance(env, dict):
        return copy.deepcopy(env)
    if isinstance(env, list):
        parsed: Dict[str, Any] = {}
        for item in env:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                parsed[key] = value
            elif isinstance(item, str):
                parsed[item] = ""
            else:
                raise ConfigError(f"Unsupported env list item type: {type(item)}")
        return parsed
    raise ConfigError(f"Unsupported env type: {type(env)}")


def merge_env(base_env: Any, model_env: Any) -> Dict[str, Any]:
    merged = env_to_dict(base_env)
    merged.update(env_to_dict(model_env))
    return merged


def build_provider_index(providers: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    defaults_per_type: Dict[str, int] = {}

    for provider in providers:
        p_type = provider.get("type")
        p_subtype = provider.get("subtype")
        if not p_type or not p_subtype:
            raise ConfigError("Each provider must define 'type' and 'subtype'.")

        key = (p_type, p_subtype)
        if key in index:
            raise ConfigError(f"Duplicate provider definition for type+subtype: {key}")

        index[key] = provider
        defaults_per_type[p_type] = defaults_per_type.get(p_type, 0) + (1 if p_subtype == "default" else 0)

    for p_type, default_count in defaults_per_type.items():
        if default_count > 1:
            raise ConfigError(f"Provider type '{p_type}' defines more than one default subtype.")

    return index


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
        cycle = " -> ".join([f"{t}/{s}" for t, s in stack + [key]])
        raise ConfigError(f"Provider inheritance cycle detected: {cycle}")

    provider = provider_index.get(key)
    if provider is None:
        raise ConfigError(f"Provider not found for model reference: type={provider_type}, subtype={provider_subtype}")

    stack.append(key)
    resolved: Dict[str, Any] = {}

    parent_subtype = provider.get("inheritsfrom")
    if parent_subtype:
        resolved = resolve_provider_parameters(provider_index, provider_type, parent_subtype, memo, stack)

    resolved = deep_merge(resolved, provider.get("parameters", {}))
    stack.pop()

    memo[key] = copy.deepcopy(resolved)
    return copy.deepcopy(resolved)


def collapse_models(config: Dict[str, Any]) -> Dict[str, Any]:
    providers = config.get("providers")
    models = config.get("models")

    if not isinstance(providers, list):
        raise ConfigError("Top-level key 'providers' must be a list.")
    if not isinstance(models, list):
        raise ConfigError("Top-level key 'models' must be a list.")

    provider_index = build_provider_index(providers)
    provider_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    collapsed: List[Dict[str, Any]] = []

    for model in models:
        if not isinstance(model, dict):
            raise ConfigError("Each model entry must be a mapping/object.")

        name = model.get("name")
        if not name:
            raise ConfigError("Each model must define 'name'.")

        provider_ref = model.get("provider")
        if not isinstance(provider_ref, dict):
            raise ConfigError(f"Model '{name}' must define a provider object with type/subtype.")

        p_type = provider_ref.get("type")
        p_subtype = provider_ref.get("subtype")
        if not p_type or not p_subtype:
            raise ConfigError(f"Model '{name}' provider must include both 'type' and 'subtype'.")

        resolved_parameters = resolve_provider_parameters(
            provider_index,
            p_type,
            p_subtype,
            provider_cache,
            stack=[],
        )

        model_parameters = model.get("parameters")
        if model_parameters is not None:
            resolved_parameters = deep_merge(resolved_parameters, model_parameters)

        if "env" in model:
            resolved_parameters["env"] = merge_env(resolved_parameters.get("env"), model.get("env"))

        if "command" in model:
            resolved_parameters["command"] = copy.deepcopy(model.get("command"))

        flattened: Dict[str, Any] = {
            "name": name,
        }

        if "modelname" in model:
            flattened["modelname"] = model["modelname"]

        flattened = deep_merge(flattened, resolved_parameters)

        for key, value in model.items():
            if key in {"provider", "parameters", "env", "command", "name", "modelname"}:
                continue
            flattened[key] = copy.deepcopy(value)

        collapsed.append(flattened)

    return {"model": collapsed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse models.yml provider inheritance into fully expanded model entries."
    )
    parser.add_argument("input", type=Path, help="Path to input models.yml")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("models.collapsed.yml"),
        help="Path to output collapsed YAML (default: models.collapsed.yml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ConfigError("Input YAML root must be a mapping/object.")

    collapsed = collapse_models(data)

    with args.output.open("w", encoding="utf-8") as f:
        yaml.dump(
            collapsed,
            f,
            Dumper=LiteralStringDumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    print(f"Wrote collapsed config to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)