# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-commit hook: enforce that non-example resource server configs contain
required metadata fields (description, value) so the README training table
never shows placeholder dashes."""

import sys
from pathlib import Path

import yaml


REQUIRED_FIELDS = ("description", "value")


def find_resource_server_block(data: dict) -> tuple[str | None, dict | None]:
    """Walk the YAML structure to locate the inner resource server config dict.

    Expected shape:
        <instance_name>:
          resources_servers:
            <server_name>:
              domain: ...
              description: ...
              value: ...

    Returns (server_name, config_dict) or (None, None) if not found.
    """
    if not isinstance(data, dict):
        return None, None

    for top_value in data.values():
        if not isinstance(top_value, dict):
            continue
        rs = top_value.get("resources_servers")
        if not isinstance(rs, dict):
            continue
        for server_name, server_cfg in rs.items():
            if isinstance(server_cfg, dict):
                return server_name, server_cfg

    return None, None


def is_example_server(yaml_path: Path) -> bool:
    """Example servers live under resources_servers/example_*/."""
    parts = yaml_path.parts
    try:
        idx = parts.index("resources_servers")
        server_dir = parts[idx + 1]
        return server_dir.startswith("example_")
    except (ValueError, IndexError):
        return False


def validate(yaml_path: Path) -> list[str]:
    """Return a list of error messages (empty means valid)."""
    if is_example_server(yaml_path):
        return []

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    server_name, server_cfg = find_resource_server_block(data)
    if server_cfg is None:
        return []

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        val = server_cfg.get(field)
        if not val or (isinstance(val, str) and not val.strip()):
            errors.append(
                f"{yaml_path}: resource server '{server_name}' is missing required field '{field}'. "
                f"Please add a non-empty '{field}' under the resource server config block."
            )

    return errors


def main() -> int:
    changed_files = sys.argv[1:]
    if not changed_files:
        return 0

    all_errors: list[str] = []
    for filepath in changed_files:
        all_errors.extend(validate(Path(filepath)))

    if all_errors:
        sys.stderr.write("[pre-commit] Resource server config validation failed:\n")
        for err in all_errors:
            sys.stderr.write(f"  - {err}\n")
        sys.stderr.write(
            "\nEvery non-example resource server config must have 'description' and 'value' fields.\n"
            "These populate the training table in README.md.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
