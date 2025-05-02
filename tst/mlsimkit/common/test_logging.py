# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import ANY

import logging
import yaml

from mlsimkit.common.schema.logging import LogConfig, Level
from mlsimkit.common.logging import configure_logging


@pytest.fixture
def mock_basicConfig(mocker):
    return mocker.patch("logging.basicConfig")


def test_configure_logging_with_level(mock_basicConfig):
    configure_logging(LogConfig(use_config_file=False, level=Level.DEBUG))
    mock_basicConfig.assert_called_with(format=ANY, level=logging.DEBUG)


def test_configure_logging_with_none_as_level(mock_basicConfig):
    configure_logging(LogConfig(use_config_file=False, level=None))
    mock_basicConfig.assert_called_with(format=ANY, level=logging.INFO)


@pytest.fixture
def mock_dictConfig(mocker):
    return mocker.patch("logging.config.dictConfig")


@pytest.fixture
def yaml_test_helper(tmp_path, mock_dictConfig):
    """Fixture to help with yaml config file tests"""

    class YAMLTestHelper:
        def __init__(self):
            self.yaml_file = tmp_path / "logging.yaml"
            tmp_path.chmod(0o777)
            self.mock_dictConfig = mock_dictConfig

        def configure_logging(self, config_yaml_str, level=None, prefix_dir=None):
            config_dict = yaml.safe_load(config_yaml_str)
            self.yaml_file.write_text(yaml.dump(config_dict))
            configure_logging(LogConfig(level=level, prefix_dir=prefix_dir, config_file=self.yaml_file))
            self.mock_dictConfig.assert_called_with(ANY)

            actual = self.mock_dictConfig.call_args[0][0]
            return actual

    return YAMLTestHelper()


class TestWithYamlFile:
    def test_custom_root_level(self, yaml_test_helper):
        s = """
        root:
          level: NOTSET
          handlers: [console]
        """
        actual = yaml_test_helper.configure_logging(s)
        assert actual["root"]["level"] == logging.NOTSET

    def test_default_root_level(self, yaml_test_helper):
        s = """
        root:
            handler: [console]
        """
        actual = yaml_test_helper.configure_logging(s)
        assert actual["root"]["level"] == logging.WARNING

    def test_root_level_overridden(self, yaml_test_helper):
        s = """
        root:
            level: NOTSET
            handler: [console]
        """
        actual = yaml_test_helper.configure_logging(s, level="ERROR")
        assert actual["root"]["level"] == logging.ERROR

    def test_prefix_dir_added_to_handlers(self, yaml_test_helper, tmp_path):
        s = """
        handlers:
            file:
                filename: test.log  
                class: mlsimkit.common.logging.FileHandler
        """
        prefix_dir = tmp_path / "logs"
        actual = yaml_test_helper.configure_logging(s, prefix_dir=prefix_dir)
        assert actual["handlers"]["file"]["filename"] == prefix_dir / "test.log"

    def test_console_level_overridden(self, yaml_test_helper, tmp_path):
        s = """
        handlers:
            console:
                class: logging.StreamHandler
                level: NOTSET
        """
        actual = yaml_test_helper.configure_logging(s, level="ERROR")
        assert actual["handlers"]["console"]["level"] == logging.ERROR
