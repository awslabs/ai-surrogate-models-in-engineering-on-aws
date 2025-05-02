# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
import click
from pathlib import Path
from typing import Dict, Callable, List, Any, Optional


class UniqueKeyLoader(yaml.SafeLoader):
    """
    Check for duplicate keys when loading YAML. Treats '-' and '_' interchangeably
    to support CLI and config formats.

    Usage:

        config = yaml.load(f, Loader=UniqueKeyLoader)

    """

    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key {key!r} found in YAML.")
            if key.replace("_", "-") in mapping or key.replace("-", "_") in mapping:
                raise ValueError(
                    f"Duplicate key {key!r}/{key.replace('_', '_')!r} found in YAML ('-' and '_' are interchangeable)"
                )
            mapping.add(key)
        return super().construct_mapping(node, deep)


def get_nested_value(
    data: Dict[str, Any], keys: List[str], default: Optional[Any] = None
) -> tuple[Optional[Any], bool]:
    """
    Recursively retrieves the value from a nested dictionary based on the provided keys.

    Args:
        data (dict): The dictionary to retrieve the value from.
        keys (list): A list of keys representing the nested path to the target value.

    Returns:
        A tuple containing:
            - The value at the specified nested path, or None if the path doesn't exist.
            - A boolean indicating whether the key was found or not.
    """
    if not keys:
        return data, True

    key = keys[0]
    keys_remaining = keys[1:]

    if key not in data:
        return default, False

    value, found = get_nested_value(data[key], keys_remaining, default)
    return value, found


def set_nested_value(data, keys, value):
    """
    Recursively sets the value in a nested dictionary based on the provided keys.

    Args:
        data (dict): The dictionary to modify.
        keys (list): A list of keys representing the nested path to the target value.
        value: The value to set.

    Returns:
        The modified dictionary.
    """
    key = keys[0]
    keys_remaining = keys[1:]

    if not keys_remaining:
        data[key] = value
    else:
        if key not in data:
            data[key] = {}
        data[key] = set_nested_value(data[key], keys_remaining, value)

    return data


def is_nested_ref(value: str, keys: List[str] = None) -> bool:
    """
    Check if a value is a nested reference (a string with a special '+' prefix).

    Args:
        value (str): The string value to check.
        keys (List[str]): The list of keys leading to the nested value

    Returns:
        bool: True if the value is a nested reference, False otherwise.
    """
    return isinstance(value, str) and value.startswith("+")


def _make_nested_dict(flat_dict: Dict[str, Any], delimiter: str = "__") -> Dict[str, dict]:
    """Construct a nested dictionary from a flat dictionary with delimited keys.

    Args:
        flat_dict: Flat dictionary with delimited key strings and any value types
        delimiter: Delimiter that separates each key part

    Returns:
        Nested dictionary
    """
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(delimiter)
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested


def load_yaml(root: Path, refvalue: str, keys: List[str]) -> Dict[str, Any]:
    """
    Load the contents of a YAML file referenced by a nested reference.

    Args:
        root (Path): The root directory of the configuration file.
        refvalue (str): The nested reference string (starting with '+').
        keys (List[str]): The list of keys leading to the nested reference.

    Returns:
        Dict[str, Any]: The loaded YAML data as a dictionary.

    Raises:
        ValueError: If the referenced YAML file is not found.
    """
    refpath = Path(refvalue[1:]) if is_nested_ref(refvalue) else Path(refvalue)
    if len(refpath.parents) >= 2:
        # folders were specified, so use path relative to the base .yaml path
        file_path = root / refpath
    else:
        # only then filename was specified, so use the keys as the relative path
        file_path = root / Path.joinpath(Path(), *keys, refpath)

    file_path = file_path.with_suffix(".yaml")

    if not file_path.is_file():
        raise ValueError(f"File '{file_path}' does not exist")

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_dict(
    d: Dict[str, Any],
    value_filter: Callable[[Any, List[str]], bool] = is_nested_ref,
    callback: Callable[[Any, List[str]], Any] = load_yaml,
) -> Dict[str, Any]:
    """
    Process a dictionary by applying a filter and callback to its values.

    Args:
        d (Dict[str, Any]): The dictionary to process.
        value_filter (Callable[[Any], bool], optional): A function that takes a value and returns True
            if the value is considered "special". Defaults to `is_nested_ref`.
        callback (Callable[[Any, List[str]], Any], optional): A function that will be called for each
            "special" value encountered in the dictionary. Defaults to `load_yaml`.

    Returns:
        Dict[str, Any]: A new dictionary with processed values.
    """

    def process_recursive(d: Dict[str, Any], keys: List[str] = []) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in d.items():
            current_keys = keys + [key]
            value = process_value(value, current_keys)
            result[key] = value
        return result

    # nosemgrep: useless-inner-function
    def process_value(value: Any, keys: List[str]) -> Any:
        if isinstance(value, dict):
            return process_recursive(value, keys)
        elif value_filter and value_filter(value, keys):
            if callback:
                return callback(value, keys)
        return value

    return process_recursive(d)


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    In case of conflicting keys, the values from `dict2` take precedence over `dict1`.

    Args:
        dict1 (Dict[str, Any]): The first dictionary.
        dict2 (Dict[str, Any]): The second dictionary.

    Returns:
        Dict[str, Any]: A new dictionary containing the merged values.
    """

    def merge_recursive(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in d1.items():
            if key in d2:
                if isinstance(value, dict) and isinstance(d2[key], dict):
                    result[key] = merge_recursive(value, d2[key])
                else:
                    result[key] = d2[key]
            else:
                result[key] = value

        for key, value in d2.items():
            if key not in result:
                result[key] = value

        return result

    return merge_recursive(dict1, dict2)


def merge_configs(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    value_filter: Callable[[Any, List[str]], bool] = is_nested_ref,
    callback: Callable[[Any, List[str]], Any] = load_yaml,
) -> Dict[str, Any]:
    """
    Merge two dictionaries after processing them with a filter and callback.

    Args:
        dict1 (Dict[str, Any]): The first dictionary.
        dict2 (Dict[str, Any]): The second dictionary.
        value_filter (Callable[[Any], bool], optional): A function that takes a value and returns True
            if the value is considered "special". Defaults to `is_nested_ref`.
        callback (Callable[[Any, List[str]], Any], optional): A function that will be called for each
            "special" value encountered in the dictionaries. Defaults to `load_yaml`.

    Returns:
        Dict[str, Any]: A new dictionary containing the merged and processed values.
    """
    processed_dict1 = process_dict(dict1, value_filter, callback)
    processed_dict2 = process_dict(dict2, value_filter, callback)
    return merge_dicts(processed_dict1, processed_dict2)


def merge_yaml_refs(
    config: Dict[str, Any],
    root_path: Optional[Path],
    yaml_ref_options: Dict[str, click.Option],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge YAML file references into the configuration dictionary.

    This function processes YAML file references specified as command-line options
    and merges their contents into the provided configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary to be updated with YAML file references.
        root_path (Optional[Path]): The path to the base configuration file, or None if not provided.
        yaml_ref_options (Dict[str, click.Option]): A dictionary mapping option names to Click options
            representing YAML file references.
        params (Dict[str, Any]): A dictionary containing the command-line parameters, including YAML file references.

    Returns:
        Dict[str, Any]: A new dictionary containing the merged configuration with YAML file references.

    Raises:
        click.BadParameter: If a relative YAML reference is provided without a base configuration file,
            or if the referenced YAML file does not exist.

    Notes:
        - The function expects YAML file references to be prefixed with '+' in the command-line parameters.
        - Relative YAML file paths are resolved with respect to the base configuration file's directory.
        - If a parameter is specified both in a YAML file and on the command line, the command-line value takes precedence.
    """
    merged_config = config.copy()

    user_specified_options = list(params.keys())
    remaining_keys = [k for k in yaml_ref_options.keys() if k not in user_specified_options]

    for key in user_specified_options + remaining_keys:
        opt = yaml_ref_options.get(key, None)
        if not opt:
            continue

        p = params.get(opt.name, None)
        if not p:
            continue

        if is_nested_ref(p) and not root_path:
            raise click.BadParameter(
                f"Relative YAML reference ('{p}') requires a base config (--config)", param=opt
            )

        try:
            option_keys = opt.name.split("__")
            value = load_yaml(root_path.parent, p, option_keys)
            merged_config = set_nested_value(merged_config, option_keys, value)
        except ValueError as e:
            raise click.BadParameter(*e.args, param=opt)

    return merged_config


def replace_chars_in_keys(
    data: Dict[str, Any],
    chars_to_replace: str = "-",
    replacement_char: str = "_",
) -> Dict[str, Any]:
    """
    Recursively replaces specified characters with a replacement character in the keys of a dictionary.

    Args:
        data (Dict[str, Any]): The input dictionary.
        chars_to_replace (str): The characters to replace in the keys (default: "-").
        replacement_char (str): The character to use as the replacement (default: "_").

    Returns:
        Dict[str, Any]: A new dictionary with the keys updated.
    """

    def replace_keys_recursive(d):
        result = {}
        for key, value in d.items():
            new_key = key.replace(chars_to_replace, replacement_char)
            if isinstance(value, dict):
                result[new_key] = replace_keys_recursive(value)
            else:
                result[new_key] = value
        return result

    return replace_keys_recursive(data)


def remap_keys(flat_dict, key_map, delimiter="."):
    """
    Remap the keys in a flat dictionary based on a key mapping.

    Args:
        flat_dict (Dict[str, Any]): The flat dictionary to remap.
        key_map (Dict[str, str]): A mapping of old key paths to new key paths.
        delimiter (str, optional): The delimiter used to separate key parts. Defaults to '.'.

    Returns:
        Dict[str, Any]: A new dictionary with the keys remapped according to the provided key mapping.
    """
    remapped_dict = {}
    for key, value in flat_dict.items():
        new_key = key
        for old_key_path, new_key_path in sorted(key_map.items(), key=lambda x: len(x[0]), reverse=True):
            if key == old_key_path:
                new_key = new_key_path
                break
            elif key.startswith(old_key_path + delimiter):
                new_key = new_key_path + delimiter + key[len(old_key_path) + len(delimiter) :]
                break
        remapped_dict[new_key] = value
    return remapped_dict


def flatten_dict(d, parent_key="", delimiter="."):
    """
    Flatten a nested dictionary by combining nested keys with a delimiter.

    Args:
        d (Dict[str, Any]): The nested dictionary to flatten.
        parent_key (str, optional): The parent key prefix for nested keys. Defaults to ''.
        delimiter (str, optional): The delimiter used to separate nested keys. Defaults to '.'.

    Returns:
        Dict[str, Any]: A flat dictionary with keys representing the nested structure.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + delimiter + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, delimiter).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict, delimiter="."):
    """
    Unflatten a flat dictionary with delimited keys into a nested dictionary.

    Args:
        flat_dict (Dict[str, Any]): The flat dictionary to unflatten.
        delimiter (str, optional): The delimiter used to separate nested keys. Defaults to '.'.

    Returns:
        Dict[str, Any]: A nested dictionary with the original key structure.
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(delimiter)
        current = nested_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return nested_dict


def move_keys(d, key_map, delimiter="."):
    """
    Move and remap keys in a nested dictionary based on a key mapping.

    This function takes a nested dictionary, a mapping of old keys to new keys, and an optional delimiter.
    It flattens the dictionary, remaps the keys according to the provided mapping, and then unflattens the
    dictionary to restore its nested structure with the remapped keys.

    Args:
        d (Dict[str, Any]): The nested dictionary to remap.
        key_map (Dict[str, str]): A mapping of old key paths to new key paths.
        delimiter (str, optional): The delimiter used to separate key parts in the flattened dictionary.
            Defaults to '.'.

    Returns:
        Dict[str, Any]: A new dictionary with the keys remapped according to the provided key mapping.
    """
    flat_dict = flatten_dict(d, delimiter=delimiter)
    remapped_flat_dict = remap_keys(flat_dict, key_map, delimiter=delimiter)
    remapped_dict = unflatten_dict(remapped_flat_dict, delimiter=delimiter)
    return remapped_dict
