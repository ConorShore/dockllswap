# dockllswap

This repository generates two runtime configs from a single source of truth:

- `models.yml` (models + provider inheritance)
- `generator.config.yml` (output file names, headers/footers, and small render options)

Generated outputs:

- `docker-compose.models.yml`
- `config/llama-swap.yaml`

## How generation works

`models.yml` contains:

- `providers`: provider definitions with deep inheritance via `inheritsfrom`
- `models`: model entries that reference provider `type` + `subtype` and can override inherited fields

`scripts/generate_from_models.py`:

1. Resolves provider inheritance recursively
2. Merges model overrides on top of provider parameters
3. Writes compose + llama-swap outputs using templates from `generator.config.yml`

## Generate files

Run:

```bash
python scripts/generate_from_models.py generator.config.yml
```

That command updates:

- `docker-compose.models.yml`
- `config/llama-swap.yaml`

## Config reference (`generator.config.yml`)

Top-level keys:

- `models_file`: path to source models file (normally `models.yml`)
- `compose`: settings for generated compose file
- `swap`: settings for generated llama-swap file

### `compose`

Common keys:

- `output_file`: output compose path
- `container_name_overrides`: optional per-model container name overrides
- `env_remove_keys`: optional env vars to drop from rendered environment
- `env_add_by_model`: optional per-model env additions
- `env_value_overrides`: optional env value normalization
- `verbose_healthcheck`: healthcheck block appended to each generated service
- `header`: literal text written before generated services
- `footer`: literal text written after generated services

### `swap`

Common keys:

- `output_file`: output llama-swap path
- `cmd_up_macro`: macro name for start commands (default behavior renders `${<macro>} <model-name>`)
- `cmd_down_macro`: macro name for stop commands (default behavior renders `${<macro>} <model-name>`)
- `header`: literal text written before generated models
- `footer`: literal text written after generated models

## Notes

- Model order in outputs follows the order of models in `models.yml`.
- Compose rendering emits explicit service fields (no YAML anchors/merge keys).
- If a provider defines `build` with an embedded `image`, generated compose emits them as sibling keys (`build` and `image`) for valid compose structure.
