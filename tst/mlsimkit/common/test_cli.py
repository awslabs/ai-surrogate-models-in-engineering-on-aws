# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import click
import importlib.resources  # nosemgrep: python37-compatibility-importlib2

from click.testing import CliRunner

from pydantic import Field
from typing import Optional, Union, List, Dict
from collections import namedtuple

from mlsimkit.common.cli import (
    BaseGroup,
    BaseCommand,
    BaseModel,
    PydanticOptions,
    Option,
    ListParamType,
    ResourcePath,
    _cls_map_from_params,
    _check_duplicate_option_names,
    _parse_param_shorthand_str,
    options,
    options_from_schema,
    options_from_schema_shorthand,
    program,
)

from mlsimkit.common.config import _make_nested_dict

#
# Models used for testing across all test suites
#


class ModelA(BaseModel):
    foo: int
    bar: str = "default"


class NestedModel(BaseModel):
    baz: float


class ModelB(BaseModel):
    nested: NestedModel


class ModelC(BaseModel):
    foo: Optional[str] = None
    bar: Optional[str]


class ModelD(BaseModel):
    foo: Union[int, str]


class ModelE(BaseModel):
    myflag: bool = True


class ModelF(BaseModel):
    ints: List[int] = []
    floats: List[float] = []
    strings: List[str] = []


class DataFile(BaseModel):
    name: str = Field(..., min_length=1)  # non-empty
    file_glob: str = Field(..., min_length=1)  # non-empty
    columns: List[str] = Field(..., min_length=1)  # non-empty
    indices: List[int] = Field([], min_length=1)  # not required, must be non-empty
    floats: List[float] = Field([], min_length=1)  # not required, must be non-empty


def assert_option(option, internal_name, external_name, cls):
    assert isinstance(option, Option)
    assert option.name == internal_name
    assert option.opts == [external_name]
    assert option.data_cls == cls


class TestPydanticOptions:
    def test_standard_fields(self):
        options = PydanticOptions.from_model(ModelA, "modela")
        assert len(options) == 2
        assert_option(options[0], "modela__foo", "--foo", ModelA)
        assert_option(options[1], "modela__bar", "--bar", ModelA)

    def test_nested_fields_without_prefix(self):
        options = PydanticOptions.from_model(ModelB, "modelb", name_prefix=None)
        assert len(options) == 1
        assert_option(options[0], "modelb__nested__baz", "--nested.baz", NestedModel)

    def test_nested_fields_with_prefix(self):
        options = PydanticOptions.from_model(ModelB, "modelb", name_prefix="modelb")
        assert len(options) == 1
        assert_option(options[0], "modelb__nested__baz", "--modelb.nested.baz", NestedModel)

    def test_optional_field(self):
        options = PydanticOptions.from_model(ModelC, "modelc")
        assert_option(options[0], "modelc__foo", "--foo", ModelC)
        assert_option(options[1], "modelc__bar", "--bar", ModelC)
        assert options[0].required is False
        assert options[1].required is True

    def test_union_not_supported(self):
        with pytest.raises(click.BadParameter):
            PydanticOptions.from_model(ModelD, "modeld")

    def test_default_value(self):
        options = PydanticOptions.from_model(ModelA, "modela")
        bar_option = [o for o in options if o.opts == ["--bar"]][0]
        assert bar_option.default == ModelA.model_fields["bar"].default

    def test_boolean_flag(self):
        options = PydanticOptions.from_model(ModelE, "modele")
        assert_option(options[0], "modele__myflag", "--myflag", ModelE)
        assert options[0].secondary_opts == ["--no-myflag"]

    def test_boolean_flag_with_prefix(self):
        options = PydanticOptions.from_model(ModelE, "modele", name_prefix="prefix")
        assert_option(options[0], "modele__myflag", "--prefix.myflag", ModelE)
        assert options[0].secondary_opts == ["--prefix.no-myflag"]

    def test_help_group(self):
        options = PydanticOptions._yield_options(ModelA, "modela", help_group="Model A Options")
        assert all(o.help_group == "Model A Options" for o in options)


class TestOptionsFromSchema:
    # Test set covers decorator behavior only because TestPydanticOptions covers the rest of the logic
    def test_decorator(self):
        @click.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(modela: ModelA):
            assert modela.foo == 5
            assert modela.bar == "default"

        result = CliRunner().invoke(cmd, ["--foo", 5])
        assert result.exit_code == 0

    def test_group(self):
        @click.command(cls=BaseGroup)
        @click.pass_context
        def group(ctx):
            pass

        # nosemgrep: useless-inner-function
        @group.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            assert modela.foo == 5
            assert modela.bar == "default"

        result = CliRunner().invoke(group, ["cmd", "--foo", 5])
        assert result.exit_code == 0

    def test_duplicate_names_raises(self):
        with pytest.raises(click.BadParameter):
            # nosemgrep: useless-inner-function
            @click.command(cls=BaseCommand)
            @options_from_schema(ModelA, "modela_1")
            @options_from_schema(ModelA, "modela_2")
            def cmd(modela_1: ModelA, modela_2: ModelA):
                pass

    def test_duplicate_names_with_prefixes_works(self):
        @click.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela_1")
        @options_from_schema(ModelA, "modela_2", prefix="prefix")
        def cmd(modela_1: ModelA, modela_2: ModelA):
            assert modela_1.foo == 5
            assert modela_2.foo == 6

        result = CliRunner().invoke(cmd, ["--foo", 5, "--prefix.foo", 6])
        assert result.exit_code == 0

    def test_list_param_type(self):
        @click.command()
        @click.option("--ints", type=ListParamType(int, delimiter=":"))
        @click.option("--floats", type=ListParamType(float))
        @click.option("--strings", type=ListParamType(str))
        def cmd(ints: List[int], floats: List[float], strings: List[str]):
            assert ints == [1, 2, 3]
            assert floats == [1.1, 2.2, 3.3]
            assert strings == ["a", "b", "c"]

        result = CliRunner().invoke(cmd, ["--ints", "1:2:3", "--floats", "1.1,2.2,3.3", "--strings", "a,b,c"])
        assert result.exit_code == 0

    def test_list_from_schema(self):
        @click.command(cls=BaseCommand)
        @options_from_schema(ModelF, "model")
        def cmd(model: ModelF):
            assert model.ints == [1, 2, 3]
            assert model.floats == [1.1, 2.2, 3.3]
            assert model.strings == ["a", "b", "c"]

        result = CliRunner().invoke(cmd, ["--ints", "1,2,3", "--floats", "1.1,2.2,3.3", "--strings", "a,b,c"])
        assert result.exit_code == 0

    def test_field_order_is_correct(self):
        @click.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(modela: ModelA):
            pass

        result = CliRunner().invoke(cmd, ["--help"])
        assert result.exit_code == 0

        foo_help = "  --foo INTEGER  [required]"
        bar_help = "  --bar TEXT"

        lines = result.output.split("\n")
        assert foo_help in lines
        assert bar_help in lines
        assert lines.index(foo_help) < lines.index(bar_help)


class TestProgramWithUseDebugMode:
    def test_debug_mode_default(self):
        @program(name="TestProgram", version="0.1", use_debug_mode=True)
        def myprogram(ctx, debug_mode):
            assert not debug_mode
            ctx.obj["debug-mode"] = debug_mode

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        def cmd(ctx):
            assert "debug-mode" in ctx.obj
            assert not ctx.obj["debug-mode"]

        result = CliRunner().invoke(myprogram, ["cmd"])
        assert result.exit_code == 0

    def test_debug_mode_false(self):
        @program(name="TestProgram", version="0.1", use_debug_mode=True)
        def myprogram(ctx, debug_mode):
            assert not debug_mode
            ctx.obj["debug-mode"] = debug_mode

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        def cmd(ctx):
            assert "debug-mode" in ctx.obj
            assert not ctx.obj["debug-mode"]
            pass

        result = CliRunner().invoke(myprogram, ["--no-debug", "cmd"])
        assert result.exit_code == 0

    def test_debug_mode_true(self):
        @program(name="TestProgram", version="0.1", use_debug_mode=True)
        def myprogram(ctx, debug_mode):
            assert debug_mode
            ctx.obj["debug-mode"] = debug_mode

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        def cmd(ctx):
            assert "debug-mode" in ctx.obj
            assert ctx.obj["debug-mode"]

        result = CliRunner().invoke(myprogram, ["--debug", "cmd"])
        assert result.exit_code == 0

    def test_debug_mode_true_handles_non_click_exception(self):
        @program(name="TestProgram", version="0.1", use_debug_mode=True)
        def myprogram(ctx, debug_mode):
            ctx.obj["debug-mode"] = debug_mode

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        def cmd(ctx):
            raise RuntimeError("uh oh")

        result = CliRunner().invoke(myprogram, ["--debug", "cmd"])
        assert isinstance(result.exception, RuntimeError)
        assert result.exit_code == 1

    def test_debug_mode_false_handles_non_click_exception(self):
        @program(name="TestProgram", version="0.1", use_debug_mode=True)
        def myprogram(ctx, debug_mode):
            ctx.obj["debug-mode"] = debug_mode

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        def cmd(ctx):
            raise RuntimeError("uh oh")

        result = CliRunner().invoke(myprogram, ["--no-debug", "cmd"])
        assert isinstance(result.exception, SystemExit)
        assert result.exit_code == 2


class TestProgramWithUseConfigFile:
    # This test set covers config behavior because TestPydanticOptions and TestOptionsFromSchema cover the rest of the logic
    def test_no_config_file_passed(self):
        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            assert modela.foo == 5
            assert modela.bar == "default"

        result = CliRunner().invoke(myprogram, ["cmd", "--foo", 5])
        assert result.exit_code == 0

    def test_config_file_and_no_cmdline(self, tmp_path):
        temp_config_file = tmp_path / "temp_config.yaml"
        temp_config_file.write_text(
            """
            cmd:
              modela:
                foo: 123
                bar: hello
            """
        )

        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            assert modela.foo == 123
            assert modela.bar == "hello"

        result = CliRunner().invoke(myprogram, ["--config", temp_config_file, "cmd"])
        assert result.exit_code == 0

    def test_cmdline_override(self, tmp_path):
        temp_config_file = tmp_path / "temp_config.yaml"
        temp_config_file.write_text(
            """
            cmd:
              modela:
                foo: 123
                bar: hello
            """
        )

        # nosemgrep: useless-inner-function
        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            assert modela.foo == 456
            assert modela.bar == "world"

        result = CliRunner().invoke(
            myprogram, ["--config", temp_config_file, "cmd", "--foo", 456, "--bar", "world"]
        )
        assert result.exit_code == 0

    def test_missing_required_field_expect_error_exit_code(self, tmp_path):
        temp_config_file = tmp_path / "temp_config.yaml"
        temp_config_file.write_text(
            """
            cmd:
              modela:
                bar: hello
            """
        )

        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            pass

        result = CliRunner().invoke(myprogram, ["--config", temp_config_file, "cmd"])
        assert result.exit_code == 2

    def test_missing_required_field_but_on_cmdline(self, tmp_path):
        temp_config_file = tmp_path / "temp_config.yaml"
        temp_config_file.write_text(
            """
            cmd:
              modela:
                bar: hello
            """
        )

        # nosemgrep: useless-inner-function
        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema(ModelA, "modela")
        def cmd(ctx, modela: ModelA):
            assert modela.foo == 999
            assert modela.bar == "goodbye"

        result = CliRunner().invoke(
            myprogram, ["--config", temp_config_file, "cmd", "--foo", 999, "--bar", "goodbye"]
        )
        assert result.exit_code == 0


class TestClsMapFromParams:
    def test_basic(self):
        params = [
            Option(param_decls=["--modela__param"], data_cls=ModelA),
            Option(param_decls=["--modelb__param"], data_cls=ModelB),
        ]
        assert _cls_map_from_params(params) == {"modela__param": ModelA, "modelb__param": ModelB}

    def test_no_data_cls(self):
        params = [Option(param_decls=["--modela__param"], data_cls=None)]
        assert _cls_map_from_params(params) == {"modela__param": None}

    def test_key_without_delimiter(self):
        params = [Option(param_decls=["--model"], data_cls=ModelA)]
        assert _cls_map_from_params(params) == {"model": ModelA}

    def test_duplicate_keys(self):
        params = [
            Option(param_decls=["--modela__param"], data_cls=ModelA),
            Option(param_decls=["--modela__param"], data_cls=ModelB),
        ]
        with pytest.raises(Exception):
            _cls_map_from_params(params)

    def test_nested_keys(self):
        params = [
            Option(param_decls=["--modelb__param"], data_cls=ModelB),
            Option(param_decls=["--modelb__nested__param"], data_cls=ModelC),
        ]
        assert _cls_map_from_params(params) == {"modelb__param": ModelB, "modelb__nested__param": ModelC}


class TestMakeNestedDict:
    def test_basic(self):
        flat_dict = {"a__b": "value1", "c": "value2"}
        expected = {"a": {"b": "value1"}, "c": "value2"}
        assert _make_nested_dict(flat_dict) == expected

    def test_empty_dict(self):
        assert _make_nested_dict({}) == {}

    def test_no_nesting(self):
        flat_dict = {"a": "value1", "b": "value2"}
        expected = flat_dict
        assert _make_nested_dict(flat_dict) == expected

    def test_custom_delimiter(self):
        flat_dict = {"a.b": "value1", "c": "value2"}
        expected = {"a": {"b": "value1"}, "c": "value2"}
        assert _make_nested_dict(flat_dict, delimiter=".") == expected

    def test_multiple_levels_nesting(self):
        flat_dict = {"a__b__c": "value1", "a__b__d": "value2", "d": "value3"}
        expected = {"a": {"b": {"c": "value1", "d": "value2"}}, "d": "value3"}
        assert _make_nested_dict(flat_dict) == expected


class TestCheckDuplicateOptionNames:
    def test_no_duplicates(self):
        opts = [click.Option(param_decls=["--opt1"]), click.Option(param_decls=["--opt2"])]
        _check_duplicate_option_names(opts)

    def test_duplicate_names(self):
        opts = [click.Option(param_decls=["--opt"]), click.Option(param_decls=["--opt"])]
        with pytest.raises(click.BadParameter):
            _check_duplicate_option_names(opts)

    def test_duplicate_prefix(self):
        opts = [click.Option(param_decls=["--model-opt"]), click.Option(param_decls=["--model-opt"])]
        with pytest.raises(click.BadParameter):
            _check_duplicate_option_names(opts)

    def test_empty_options(self):
        opts = []
        _check_duplicate_option_names(opts)

    def test_duplicate_across_opts(self):
        opts = [click.Option(param_decls=["--opt1", "--opt2"]), click.Option(param_decls=["--opt2"])]
        with pytest.raises(click.BadParameter):
            _check_duplicate_option_names(opts)


class TestResourcePath:
    def generate_cases():
        TestCase = namedtuple("TestCase", ["value", "search_paths", "expected_result", "required", "exists"])
        return [
            # cases for regular file without falling back to search paths
            TestCase("existing_file.txt", [], "existing_file.txt", required=False, exists=True),
            TestCase(
                "/path/to/existing_file.txt", [], "/path/to/existing_file.txt", required=False, exists=True
            ),
            TestCase("non_existing_file.txt", [], None, required=False, exists=False),
            # cases that fallback to search paths
            TestCase(
                "sample.yaml",
                ["mlsimkit.datasets"],
                "/path/to/mlsimkit/datasets/sample.yaml",
                required=False,
                exists=False,
            ),
            TestCase("sample.yaml", ["does.not.exist"], None, required=False, exists=False),
            TestCase(
                "sample.yaml", ["does.not.exist"], click.exceptions.BadParameter, required=True, exists=False
            ),
        ]

    @pytest.mark.parametrize("test_case", generate_cases())
    def test_convert(self, test_case, monkeypatch):
        value = test_case.value
        search_paths = test_case.search_paths
        expected_result = test_case.expected_result
        required = test_case.required
        file_exists = test_case.exists

        param = click.Option(param_decls=["--path"], required=required)
        command = click.Command("test_command", params=[param])
        ctx = click.Context(command)

        resource_path = ResourcePath(search_paths=search_paths)

        # Mock os.path.exists
        with monkeypatch.context() as m:
            m.setattr(os.path, "exists", lambda path: file_exists)

            # Mock importlib.resources.files
            if search_paths:

                def mock_files(search_path):
                    return {
                        "mlsimkit": "/path/to/mlsimkit",
                    }.get(search_path, None)

                m.setattr(importlib.resources, "files", mock_files)

            expect_exception = isinstance(expected_result, type) and issubclass(
                click.exceptions.BadParameter, expected_result
            )
            if expect_exception:
                try:
                    resource_path.convert(value, param, ctx)
                except expected_result:
                    pass
                else:
                    assert False
            else:
                result = resource_path.convert(value, param, ctx)
                assert result == expected_result or result is None


class TestParseParamShorthand:
    @pytest.mark.parametrize(
        "input_value, expected_output",
        [
            (
                "name=test,file_glob=data.dat,columns=col1",
                DataFile(name="test", file_glob="data.dat", columns=["col1"]),
            ),
            # empty columns are ignored
            (
                "name=test,file_glob=data.dat,columns=col1 col2",
                DataFile(name="test", file_glob="data.dat", columns=["col1", "col2"]),
            ),
            # multi-value param order doesn't matter
            (
                "name=test,columns=col1 col2,file_glob=data.dat,",
                DataFile(name="test", file_glob="data.dat", columns=["col1", "col2"]),
            ),
            # value with a single space character is valid
            (
                "name= ,file_glob=data.dat,columns=col1",
                DataFile(name=" ", file_glob="data.dat", columns=["col1"]),
            ),
            # value with a single comma character is valid, quotechar is "|"
            (
                "|name=,|,file_glob=data.dat,columns=col1",
                DataFile(name=",", file_glob="data.dat", columns=["col1"]),
            ),
            # list with non-str value types
            (
                "name=test,file_glob=data.dat,columns=col1 col2,indices=1 2 3,floats=1.2 2.3",
                DataFile(
                    name="test",
                    file_glob="data.dat",
                    columns=["col1", "col2"],
                    indices=[1, 2, 3],
                    floats=[1.2, 2.3],
                ),
            ),
        ],
    )
    def test_convert_manifest_param_valid(self, input_value, expected_output):
        assert _parse_param_shorthand_str(input_value, DataFile) == expected_output

    @pytest.mark.parametrize(
        "input_value",
        [
            # missing key
            "file_glob=data.dat,columns=col1",
            "name=test,columns=col1",
            "name=test,file_glob=data.dat",
            "name,file_glob=data.dat,columns=col1",
            # empty values
            "name=,file_glob=data.dat,columns=col1",
            "name=test,file_glob=,columns=col1",
            "name=test,file_glob=data.dat,columns=",
            "name=test,file_glob=data.dat,columns=col1,indices=",
        ],
    )
    def test_parse_param_shorthand_str_invalid(self, input_value):
        try:
            _parse_param_shorthand_str(input_value, DataFile)
        except Exception:
            pass
        else:
            assert False

    @pytest.mark.parametrize(
        "field_type",
        [
            Dict[str, int],
            Union[str, int],
        ],
    )
    def test_unsupported_field_type(self, field_type):
        class MyModel(BaseModel):
            badfield: field_type

        with pytest.raises(click.BadParameter):
            _parse_param_shorthand_str("badfield=brg", MyModel)


class TestOptionsFromSchemaShorthand:
    """Test the options decorator that sits on-top of the param shorthand parser"""

    def test_basic(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel)
        def cmd(option: MyModel):
            assert option.name == "John Doe"
            assert option.age == 30

        result = CliRunner().invoke(cmd, ["--option", "name=John Doe,age=30"])
        assert result.exit_code == 0

    def test_basic_multiple(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel, multiple=True)
        def cmd(option: MyModel):
            assert len(option) == 2
            assert option[0].name == "Alice"
            assert option[0].age == 25
            assert option[1].name == "Bob"
            assert option[1].age == 30

        result = CliRunner().invoke(cmd, ["--option", "name=Alice,age=25", "--option", "name=Bob,age=30"])
        assert result.exit_code == 0

    def test_with_list_field_and_non_string_type(self):
        class MyModel(BaseModel):
            names: List[str]
            ages: List[int]

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel)
        def cmd(option: MyModel):
            assert option.names == ["Alice", "Bob", "Charlie"]
            assert option.ages == [25, 30, 35]

        result = CliRunner().invoke(cmd, ["--option", "names=Alice Bob Charlie, ages=25 30 35"])
        assert result.exit_code == 0

    def test_invalid_key(self):
        class MyModel(BaseModel):
            name: str

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel)
        def cmd(option: MyModel):
            pass

        result = CliRunner().invoke(cmd, ["--option", "invalid_key=value"])
        assert result.exit_code == 2
        assert "Unknown key 'invalid_key'" in result.output

    def test_duplicate_key(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel)
        def cmd(option: MyModel):
            pass

        result = CliRunner().invoke(cmd, ["--option", "name=John,name=Jane"])
        assert result.exit_code == 2
        assert "Duplicate key 'name'" in result.output

    def test_cannot_specify_type(self):
        class MyModel(BaseModel):
            name: str

        with pytest.raises(ValueError):

            @click.command()
            @options_from_schema_shorthand("--option", model=MyModel, type=str)
            def cmd(option: MyModel):
                pass

    def test_cannot_specify_callback(self):
        class MyModel(BaseModel):
            name: str

        with pytest.raises(ValueError):

            @click.command()
            @options_from_schema_shorthand(
                "--option", model=MyModel, callback=lambda ctx, param, value: value
            )
            def cmd(option: MyModel):
                pass

    def test_short_and_long_option_names(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command()
        @options_from_schema_shorthand("-o", "--option", model=MyModel)
        def cmd(option: MyModel):
            assert option.name == "John Doe"
            assert option.age == 30

        result = CliRunner().invoke(cmd, ["-o", "name=John Doe,age=30"])
        assert result.exit_code == 0

    def test_load_from_config(self, tmp_path):
        class MyModel(BaseModel):
            name: str
            age: int

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
            cmd:
              option:
                name: John Doe
                age: 30
            """
        )

        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema_shorthand("--option", model=MyModel, multiple=False)
        def cmd(ctx, option: MyModel):
            assert option.name == "John Doe"
            assert option.age == 30

        result = CliRunner().invoke(myprogram, ["--config", str(config_file), "cmd"])
        assert result.exit_code == 0

    def test_load_from_config_multiple(self, tmp_path):
        class MyModel(BaseModel):
            name: str
            age: int

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
            cmd:
              option:
                - name: Alice
                  age: 25
                - name: Bob
                  age: 30
            """
        )

        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema_shorthand("--option", model=MyModel, multiple=True)
        def cmd(ctx, option: MyModel):
            assert len(option) == 2
            assert option[0].name == "Alice"
            assert option[0].age == 25
            assert option[1].name == "Bob"
            assert option[1].age == 30

        result = CliRunner().invoke(myprogram, ["--config", str(config_file), "cmd"])
        assert result.exit_code == 0

    def test_load_from_config_unexpected_multiple(self, tmp_path):
        class MyModel(BaseModel):
            name: str
            age: int

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
            cmd:
              option:
                - name: Alice
                  age: 25
                - name: Bob
                  age: 30
            """
        )

        @program(name="TestProgram", version="0.1", use_config_file=True)
        def myprogram(ctx, config_file):
            ctx.obj["config_file"] = config_file

        # nosemgrep: useless-inner-function
        @myprogram.command(cls=BaseCommand)
        @options_from_schema_shorthand("--option", model=MyModel, multiple=False)
        def cmd(ctx, option: MyModel):
            pass

        result = CliRunner().invoke(myprogram, ["--config", str(config_file), "cmd"])
        assert result.exit_code == 2


class TestOptionsDecorator:
    """Test the top-level options decorator pass through properly"""

    def test_click_pass_thru(self):
        @click.command()
        @options("--name", type=str)
        @options("--age", type=int)
        def cmd(name: str, age: int):
            assert name == "John Doe"
            assert age == 30

        result = CliRunner().invoke(cmd, ["--name", "John Doe", "--age", 30])
        assert result.exit_code == 0

    def test_schema_shorthand(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command()
        @options("-o", "--option", type=MyModel, shorthand=True)
        def cmd(option: MyModel):
            assert option.name == "John Doe"
            assert option.age == 30

        result = CliRunner().invoke(cmd, ["-o", "name=John Doe,age=30"])
        assert result.exit_code == 0

    def test_schema_base_command_non_shorthand(self):
        class MyModel(BaseModel):
            name: str
            age: int

        @click.command(cls=BaseCommand)
        @options(type=MyModel, dest="option", shorthand=False)
        def cmd(option: MyModel):
            assert option.name == "John Doe"
            assert option.age == 30

        result = CliRunner().invoke(cmd, ["--name", "John Doe", "--age", 30])
        assert result.exit_code == 0

    def test_schema_non_shorthand_error_on_param_decls(self):
        class MyModel(BaseModel):
            name: str
            age: int

        with pytest.raises(ValueError):
            # nosemgrep: useless-inner-function
            @click.command()
            @options("--params_decls_not_allowed", type=MyModel, dest="option", shorthand=False)
            def cmd(option: MyModel):
                pass
