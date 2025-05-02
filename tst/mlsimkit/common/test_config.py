# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import click
import yaml

from pathlib import Path
from mlsimkit.common.config import (
    merge_dicts,
    process_dict,
    merge_configs,
    is_nested_ref,
    load_yaml,
    merge_yaml_refs,
    move_keys,
)


# Tests for merge_dicts
def test_merge_dicts_without_conflicts():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    expected = {"a": 1, "b": 2, "c": 3, "d": 4}

    result = merge_dicts(dict1, dict2)
    assert result == expected


def test_merge_dicts_with_conflicts():
    dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
    dict2 = {"b": {"d": 4, "e": 5}, "f": 6}
    expected = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}

    result = merge_dicts(dict1, dict2)
    assert result == expected


# Tests for process_dict
def test_process_dict_without_filter_and_callback():
    d = {"a": 1, "b": 2, "c": {"d": 3}}
    expected = {"a": 1, "b": 2, "c": {"d": 3}}

    result = process_dict(d)
    assert result == expected


def test_process_dict_with_filter_and_callback():
    d = {"a": 1, "b": [2, 3], "c": {"d": 4, "e": [5, 6]}}

    def value_filter(value, keys):
        return isinstance(value, list)

    def callback(value, keys):
        return sum(value)

    expected = {"a": 1, "b": 5, "c": {"d": 4, "e": 11}}

    result = process_dict(d, value_filter=value_filter, callback=callback)
    assert result == expected


# Minimal tests for merge_configs
def test_merge_configs_without_filter_and_callback():
    dict1 = {"a": 1, "b": 2, "c": {"d": 4, "e": 5}}
    dict2 = {"b": 3, "c": {"e": 6, "f": 7}}
    expected = {"a": 1, "b": 3, "c": {"d": 4, "e": 6, "f": 7}}

    result = merge_configs(dict1, dict2)
    assert result == expected


def test_merge_configs_with_filter_and_callback():
    dict1 = {"a": 1, "b": [2, 3], "c": {"d": 4, "e": [5, 6]}}
    dict2 = {"b": [4, 5], "c": {"e": [7, 8], "f": 9}}

    def value_filter(value, keys):
        return isinstance(value, list)

    def callback(value, keys):
        return sum(value)

    expected = {"a": 1, "b": 9, "c": {"d": 4, "e": 15, "f": 9}}

    result = merge_configs(dict1, dict2, value_filter=value_filter, callback=callback)
    assert result == expected


#
# Mock .yaml files on disk to test load_yaml and merge_configs together
#
@pytest.fixture
def mock_yaml_files(tmp_path):
    tmp_path = tmp_path / "config_files"
    tmp_path.mkdir()

    file1_path = tmp_path / "b" / "file1.yaml"
    file1_path.parent.mkdir()
    file1_content = "file1_data: value1"
    file1_path.write_text(file1_content)

    file2_path = tmp_path / "nested" / "file2.yaml"
    file2_path.parent.mkdir()
    file2_content = "file2_data: value2"
    file2_path.write_text(file2_content)

    return tmp_path


def test_merge_configs_with_nested_refs(mock_yaml_files):
    dict1 = {"a": 1, "b": "+file1", "c": {"d": 4, "e": [5, 6]}}
    dict2 = {"c": {"e": [7, 8], "f": "+nested/file2"}, "g": 10}

    def callback(value, keys):
        config_file = Path(mock_yaml_files / "config.yaml")
        return load_yaml(config_file.parent, value, keys)

    expected = {
        "a": 1,
        "b": {"file1_data": "value1"},
        "c": {"d": 4, "e": [7, 8], "f": {"file2_data": "value2"}},
        "g": 10,
    }

    result = merge_configs(dict1, dict2, value_filter=is_nested_ref, callback=callback)
    assert result == expected


@pytest.fixture
def test(tmp_path):
    class YAMLTestHelper:
        def __init__(self):
            self.base_dir = tmp_path
            self.base_dir.chmod(0o777)

        def write(self, path, config_yaml_str):
            yaml_file = self.base_dir / path
            yaml_file.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml.safe_load(config_yaml_str), f)
            return yaml_file

        def load(self, path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        # Helper to reduce test verbiage
        def run_merge_yaml_refs(self, options, params):
            config = """
            section1: 
              ref1:
                key1: value1
            """

            file1 = """
            ref1:
              key1: file1.value1
            """

            config_file = self.write("config.yaml", config)
            self.write("section1/file1.yaml", file1)
            return merge_yaml_refs(self.load(config_file), config_file, options, params)

    return YAMLTestHelper()


class TestMergeYamlRefs:
    def test_load_yaml(self, test):
        section1 = """
        key1: value1
        """

        test.write("section1.yaml", section1)
        assert load_yaml(test.base_dir, "+section1", []) == {"key1": "value1"}
        assert load_yaml(test.base_dir, "section1", []) == {"key1": "value1"}
        assert load_yaml(test.base_dir, "section1.yaml", []) == {"key1": "value1"}

        section2 = """
        key2: value2
        """

        test.write("path/to/key/section2.yaml", section2)
        assert load_yaml(test.base_dir, "section2", ["path", "to", "key"]) == {"key2": "value2"}

    def test_merge_yaml_refs_no_options(self, test):
        config = """
        section1: 
          ref1:
            key1: value1
        """

        config_file = test.write("config.yaml", config)
        expected = yaml.safe_load(config)

        result = merge_yaml_refs(test.load(config_file), config_file, {}, {})
        assert result == expected

    def test_merge_yaml_refs_named_ref(self, test):
        options = {
            "section1": click.Option(["--section1", "section1"]),
        }
        params = {
            "section1": "+file1",
        }

        expected = yaml.safe_load(
            """
        section1:
          ref1:
            key1: file1.value1
        """
        )

        result = test.run_merge_yaml_refs(options, params)
        assert result == expected

    def test_merge_yaml_refs_file_ref(self, test):
        options = {
            "section1": click.Option(["--section1", "section1"]),
        }
        params = {
            "section1": "section1/file1.yaml",
        }

        expected = yaml.safe_load(
            """
        section1:
          ref1:
            key1: file1.value1
        """
        )

        result = test.run_merge_yaml_refs(options, params)
        assert result == expected

    def test_merge_yaml_refs_named_ref_does_not_exist(self, test):
        with pytest.raises(click.BadParameter):
            options = {
                "section1": click.Option(["--section1", "section1"]),
            }
            params = {
                "section1": "fileX",
            }
            test.run_merge_yaml_refs(options, params)

    def test_merge_yaml_refs_file_ref_does_not_exist(self, test):
        with pytest.raises(click.BadParameter):
            options = {
                "section1": click.Option(["--section1", "section1"]),
            }
            params = {
                "section1": "sectionX/file1.yaml",
            }
            test.run_merge_yaml_refs(options, params)

    def test_merge_yaml_refs_nested_option(self, test):
        config = """
        section1: 
            ref1:
              key1: value1
        """

        valueA = """
        key1: valueA
        """

        valueB = """
        key1: valueB
        """

        options = {
            "section1": click.Option(["--section1-ref1", "section1__ref1"]),
        }
        params = {
            "section1__ref1": "+valueB",
        }

        config_file = test.write("config.yaml", config)
        test.write("section1/ref1/valueA.yaml", valueA)
        test.write("section1/ref1/valueB.yaml", valueB)

        expected = yaml.safe_load(
            """
        section1:
          ref1:
            key1: valueB
        """
        )

        result = merge_yaml_refs(test.load(config_file), config_file, options, params)
        assert result == expected


class TestMoveKeys:
    @pytest.fixture
    def original_data(self):
        # fmt: off
        return {
            "level": 2, 
            "property-one": "foo", 
            "groupB": {
                "property-one": "bar", 
                "percentile": "P95"
            }
        }
        # fmt: on

    def test_move_single_key(self, original_data):
        key_mappings = {"level": "settings.level"}
        expected_output = {
            "settings": {"level": 2},
            "property-one": "foo",
            "groupB": {"property-one": "bar", "percentile": "P95"},
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output

    def test_move_multiple_keys_to_same_mapped_parent_key(self, original_data):
        key_mappings = {"level": "settings.level", "property-one": "settings.property-one"}
        expected_output = {
            "settings": {"level": 2, "property-one": "foo"},
            "groupB": {"property-one": "bar", "percentile": "P95"},
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output

    def test_move_nested_key(self, original_data):
        key_mappings = {"groupB": "settings.more_settings.groupB"}
        expected_output = {
            "level": 2,
            "property-one": "foo",
            "settings": {"more_settings": {"groupB": {"property-one": "bar", "percentile": "P95"}}},
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output

    def test_move_keys_to_nested_locations(self, original_data):
        key_mappings = {
            "level": "settings.nested.level",
            "property-one": "settings.nested-again.property-one",
        }
        expected_output = {
            "settings": {"nested": {"level": 2}, "nested-again": {"property-one": "foo"}},
            "groupB": {"property-one": "bar", "percentile": "P95"},
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output

    def test_move_and_merge_with_existing_value(self, original_data):
        original_data["settings"] = {"existing": "value"}
        key_mappings = {"groupB": "settings.more_settings.groupB"}
        expected_output = {
            "level": 2,
            "property-one": "foo",
            "settings": {
                "existing": "value",
                "more_settings": {"groupB": {"property-one": "bar", "percentile": "P95"}},
            },
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output

    def test_move_and_merge_with_existing_nested_value(self, original_data):
        original_data["settings"] = {"existing": {"nested": "value"}}
        key_mappings = {"groupB": "settings.no_groupB"}
        expected_output = {
            "level": 2,
            "property-one": "foo",
            "settings": {
                "existing": {"nested": "value"},
                "no_groupB": {"property-one": "bar", "percentile": "P95"},
            },
        }

        updated_data = move_keys(original_data, key_mappings)
        assert updated_data == expected_output
