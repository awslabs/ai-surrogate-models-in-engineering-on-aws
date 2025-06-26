# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The ``mlsimkit.common.cli`` module offers a convenient way to create rich command-line interfaces (CLIs) for Python applications with automated options generation and YAML configuration loading. It leverages the power of the `Click <https://click.palletsprojects.com/>`_ library and seamlessly integrates with `Pydantic <https://pydantic-docs.helpmanual.io/>`_ models, allowing developers to define and validate command-line options based on their data models.

The key components of this framework include:

- Command decorators for creating entry points and grouping commands
- Utilities for converting Pydantic models into Click options
- Click-compatible parameter types for handling various data types
- Opinionated custom Click classes for top-level program, commands and sub-command behaviors.

Example:

.. code-block:: python

    from pydantic import BaseModel

    class MySettings(BaseModel):
        level: int
        mode: bool

    import click
    import mlsimkit

    @mlsimkit.cli.program("tool", version="1.0", use_config_file=True)
    @mlsimkit.cli.options(MySettings, dest='settings')
    @click.option("--normal-option")
    def cli(ctx: click.Context, settings: MySettings):
        click.echo(settings.dict())

    if __name__ == "__main__":
        cli()


.. code-block:: shell

    % python3 mycli.py --help
    Usage: mycli.py [OPTIONS] COMMAND [ARGS]...

    Options:
      --version             Show the version and exit.
      --level INTEGER       [required]
      --mode / --no-mode    [required]
      --normal-option TEXT
      --help                Show this message and exit.


See :func:`options` for more API usage examples or follow the :ref:`quickstart user guide<quickstart-cli-framework>`.

"""

import os
import click
import yaml
import logging
import json
import csv
import click.decorators
import importlib.resources  # nosemgrep: python37-compatibility-importlib2
from click.core import ParameterSource
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from pydantic.v1.utils import lenient_issubclass
from pydantic._internal._model_construction import (
    ModelMetaclass,
)  # internal, see https://github.com/pydantic/pydantic/issues/6381
from typing import (
    _GenericAlias,
    Type,
    TypeVar,
    List,
    Dict,
    Set,
    Any,
    Callable,
    Union,
    Literal,
    get_origin,
    get_args,
)
from collections import OrderedDict
from collections.abc import Iterator
from io import StringIO
from enum import Enum
from inspect import isclass
from pathlib import Path

from mlsimkit.common.schema.cli import CliExtras
from . import config as conf


log = logging.getLogger(__name__)

T = TypeVar("T")


OPTION_CONFIG_FILE = click.Option(
    param_decls=["--config", "config_file"],
    type=click.Path(exists=True),
    help="Path to YAML config file",
)

OPTION_DEBUG_MODE = click.Option(
    param_decls=["--debug/--no-debug", "debug_mode"],
    is_flag=True,
    default=False,
    help="Enable debugging, such as allowing exceptions to escape programs",
)

DELIMITER = "."  # character to separate options nested in schemas


def group(**attrs):
    """A parent command for grouping of sub-commands. Ensures all sub-commands support Pydantic-based options and always passes the context.

    Args:
        attrs: Keyword arguments to pass to :class:`click.Group`.

    Returns:
        A decorator function that wraps a Click command group with the specified attributes.

    This decorator ensures that the context and Pydantic-based options are passed to the command group.
    It also sets the ``show_default`` context setting to ``True`` to display default values in the help text.
    """

    def decorator(func):
        ctx_settings = attrs.get("context_settings", {})
        ctx_settings["show_default"] = True
        attrs["context_settings"] = ctx_settings
        # attrs["invoke_without_command"] = True
        decorators = [
            click.group(cls=BaseGroup, **attrs),
            click.pass_context,
        ]
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return decorator


def program(name, version, use_config_file=False, use_debug_mode=False, **attrs):
    """Decorator to turn a function into a program. A program is a :class:`click.Group` with a name and version.

    This decorator adds a version option and the ``--config`` option (if ``use_config_file`` is True) to the command group.
    The ``--config`` option allows specifying a top-level configuration file that can be accessed by subgroups and subcommands.

    Args:
        name (str): The name of the program.
        version (str): The version string for the program.
        use_config_file (bool): Whether to enable the ``--config`` option for specifying a top-level configuration file.
        use_debug_mode (bool): Whether to enable the ``--debug/--no-debug`` option for catching exceptions from commands.
        attrs: Keyword arguments to pass to :func:`group`.

    Returns:
        A decorator function that wraps a Click command group with the specified attributes and program metadata.

    """

    def decorator(func):
        func = click.version_option(version, prog_name=name)(func)
        func = group(**attrs)(func)

        # Programs accept `--config` to allow for the conf hierarchy
        # so sub-groups/commands can assume `config_file` option exists.
        if use_config_file:
            click.decorators._param_memo(func, OPTION_CONFIG_FILE)

        if use_debug_mode:
            click.decorators._param_memo(func, OPTION_DEBUG_MODE)

        return func

    return decorator


def options(
    *param_decls: str,
    cls: type[click.Option] = None,
    **attrs: Any,
) -> Callable[[Callable], Callable]:
    """Decorator to define Click command options using Pydantic schemas.

    This decorator can be used in three ways:

    **1. Click.Option:**

    Add a standard `click.options` decorator (see `Click documentation <https://click.palletsprojects.com/en/8.1.x/options/>`__)::

        @click.command()
        @options('--level', type=int)
        @options('--mode', type=bool)
        def cmd(level, mode):
            ...

    Click options are added as a conventional command-line interface::

        $ python cmd.py --help
        Usage: cmd.py [OPTIONS]

        Options:
          --level INTEGER
          --mode BOOLEAN
          --help           Show this message and exit.

    **2. Pydantic Shorthand:**

    Add an option that accepts a shorthand string converted to a Pydantic model. See :func:`options_from_schema_shorthand` for complete usage::

        from pydantic import BaseModel

        class MySettings(BaseModel):
            level: int
            mode: bool

        @click.command()
        @options('--settings', type=MySettings, shorthand=True)
        def cmd(settings: MySettings):
            print(settings)


    The shorthand format is a comma-separate key-value string::

        $ python cmd.py --settings "level=1,mode=True"
        level=1 mode=True


    **3. Pydantic Detailed:**

    Automatically generate options for ALL fields in a Pydantic model. See :func:`options_from_schema` for complete usage::

        from pydantic import BaseModel

        class MySettings(BaseModel):
            level: int
            mode: bool

        @click.command(cls=BaseCommand)
        @options(type=MySettings, dest='settings')
        def cmd(settings: MySettings):
            ...

    The schema fields become options::

        $ python cmd.py --help
        Usage: cmd.py [OPTIONS]

        Options:
          --level INTEGER     [required]
          --mode / --no-mode  [required]
          --help              Show this message and exit.


    Args:
        *param_decls (str): The parameter names (e.g, '-o', '--option', 'option') for :func:`options_from_schema_shorthand` or click.options.
                            Alternatively, for convenience, also accepts a Pydantic model type for adding detailed options via :func:`options_from_schema`.
        **attrs: Additional keyword arguments passed to `click.options`, or :func:`options_from_schema_shorthand` or :func:`options_from_schema`. See their documentation for the available arguments.
        cls (type[click.Option], optional): The Click option class to use. Defaults to `click.Option`.

    Returns:
        Callable: A decorator function that takes a command function and returns a
            decorated version with the specified options.
    """
    # let the user specify schema as first argument Or click param declarations
    ptype = attrs.get("type", param_decls[0] if len(param_decls) >= 1 else None)
    if ptype and issubclass(ptype, BaseModel):
        # automatically use schema decorators
        if attrs.get("shorthand", False):
            attrs.pop("type", None)
            attrs.pop("shorthand", None)
            return options_from_schema_shorthand(*param_decls, **attrs, model=ptype, cls=cls)
        else:
            if param_decls and "type" in attrs or len(param_decls) > 1:
                raise ValueError(
                    f"Cannot specify parameter names when using schema, expected schema type as first argument but got '{param_decls}'"
                )
            attrs.pop("type", None)
            attrs.pop("shorthand", None)
            return options_from_schema(model=ptype, **attrs)
    else:
        # otherwise pass through to click.options
        return _option_wrapper(*param_decls, **attrs, cls=cls)


def options_from_schema_shorthand(
    *param_decls,
    cls: type[click.Option] = None,
    model: Type[BaseModel],
    delimiter=",",
    quotechar="|",
    **attrs,
):
    """Decorator to generate a Click option from a Pydantic model using shorthand notation.

    This decorator generates a single Click option that accepts a shorthand string
    representation of the Pydantic model's fields. The shorthand string should be
    in the format ``"key1=value1, key2=value2, ..., keyN=valueN"``.

    Example::

        from pydantic import BaseModel

        class MySettings(BaseModel):
            level: int
            mode: bool

        @click.command()
        @options_from_schema_shorthand('--settings', model=MySettings)
        def cmd(settings: MySettings):
            print(settings)

    From the command line::

        $ python script.py --settings "level=10, mode=true"
        level=10 mode=True

    Standard ``click.option`` parameters are supported like short names (``-o``, ``--option``, ...) and ``multiple=True`` for accepting
    more than one shorthand strings::

        @click.command()
        @options_from_schema_shorthand(
            '-s', '--setting', 'settings',
            model=MySettings, multiple=True
        )
        def cmd(settings: MySettings):
            for s in settings:
                print(s)


    From the command line::

        $ python script.py -s "level=1,mode=true" -s "level=2,mode=false"
        level=1 mode=True
        level=2 mode=False


    Pydantic List fields and non-string types are supported::

        class MyModel(BaseModel):
            names: List[str]
            ages: List[int]

        @click.command()
        @options_from_schema_shorthand("--option", model=MyModel)
        def cmd(option: MyModel):
            print(option)

    From the command line::

        $ python script.py --option "names=Alice Bob Charlie, ages=25 30 35"
        names=['Alice', 'Bob', 'Charlie'] ages=[25, 30, 35]


    A shorthand option can be loaded configuration file just like :func:`options_from_schema`,
    which is convenient when using :func:`program` ``--config`` functionality:

    .. code-block:: yaml

        # config.yaml
        option:
          names: [Alice, Bob, Charlie]
          ages: [25, 30, 35]

    With ``multiple=True``, the configuration supports lists:

    .. code-block:: yaml

        # config.yaml
        option:
          - names: [Alice, Bob, Charlie]
            ages: [25, 30, 35]
          - names: [Apple, Orange, Banana]
            ages: [1, 2, 3]
          - names: [Blue, Red, Green]
            ages: [0, 0, 0]

    And the corresponding program code::

        @mlsimkit.cli.program(name="MyProgram", version="0.1", use_config_file=True)
        def program(ctx, config_file):
            ctx.obj["config_file"] = config_file

        @program.command()
        @options_from_schema_shorthand("--option", model=MyModel, multiple=True)
        def cmd(ctx, option: MyModel):
            for opt in option:
                print(opt)


    Args:
        *param_decls (str): Parameter declarations for the Click option.
        cls (type[click.Option], optional): The Click option class to use. Defaults to
            `click.Option`.
        model (Type[BaseModel]): The Pydantic model class to generate the option for.
        **attrs: Additional keyword arguments to pass to the underlying `click.Option`.

    Returns:
        Callable: A decorator function that takes a command function and returns a
            decorated version with the specified options.

    Raises:
        ValueError: If `type` or `callback` are specified in `**attrs`, as they
            are handled automatically by this decorator.
    """
    if "type" in attrs or "callback" in attrs:
        raise ValueError("Cannot specify 'type' or 'callback' when using a shorthand option")

    return _option_wrapper(
        *param_decls,
        **attrs,
        cls=cls,
        type=ShorthandParamType(model, delimiter, quotechar),
        callback=lambda ctx, param, value: value,
    )


def options_from_schema(
    model: Type[BaseModel],
    dest: str,
    prefix: str = None,
    help_group: str = None,
    yaml_file: Union[bool, Dict] = False,
) -> Callable[[Callable], Callable]:
    """Decorator to generate Click options for all fields in a Pydantic model.

    This decorator generates Click options based on the fields of a given Pydantic
    model. The generated options can be used to parse command-line arguments and
    instantiate the model with the parsed values.

    Example::

        from pydantic import BaseModel

        class MySettings(BaseModel):
            level: int
            mode: bool

        @click.command(cls=BaseCommand)
        @options_from_schema(MySettings, dest='settings')
        def cmd(settings: MySettings):
            ...

    Args:
        model (Type[BaseModel]): The Pydantic model class to generate options for.
        dest (str): The attribute name to store the instantiated model object on the
            command context.
        prefix (str, optional): An optional prefix to add to each option name. Use
            this if duplicately named fields exist across different models used in
            the same command.
        help_group (str, optional): The help group text to display in the CLI.
        yaml_file (Union[bool, Dict], optional): Whether to add a hidden option for
            overriding the nested field from a YAML file. If set to `True`, a
            hidden option will be added with the default name. If set to a
            dictionary, it should contain the keys `name` (for the option name) and
            `hidden` (boolean indicating whether the option should be hidden).

    Returns:
        Callable: A decorator function that takes a command function and returns a
            decorated version with the specified options.

    Note:
        If multiple Pydantic models are used in the same command, any duplicate
        option names across models will be detected. Use the `prefix` argument to
        differentiate such options.
    """

    def decorator(f):
        options = PydanticOptions.from_model(
            model, dest, name_prefix=prefix, help_group=help_group, yaml_file=yaml_file
        )

        for option in reversed(options):  # reverse to original BaseModel order
            click.decorators._param_memo(f, option)  # append options to function
        _check_duplicate_option_names(f.__dict__["__click_params__"])

        # Add this Pydantic model class to the function metadata so we can
        # later instantiate instances when commands are invoked
        if isinstance(f, click.Command):
            f.cls_map.append[dest] = model
        else:
            if not hasattr(f, "__mlsimkit_cls_map__"):
                f.__mlsimkit_cls_map__ = {}
            f.__mlsimkit_cls_map__[dest] = model

        return f

    return decorator


def _option_wrapper(
    *param_decls: str,
    cls: type[click.Option] = None,
    **attrs: Any,
) -> Callable[[Callable], Callable]:
    if cls is None:
        cls = click.Option

    def decorator(f):
        click.decorators._param_memo(f, cls(param_decls, **attrs))
        return f

    return decorator


def get_command_path(ctx):
    parent_groups = [ctx.command.name]

    def traverse_parents(current_ctx):
        if current_ctx.parent:
            parent_groups.append(current_ctx.parent.info_name)
            traverse_parents(current_ctx.parent)

    traverse_parents(ctx)
    return parent_groups[::-1]  # Reverse the list to get the correct order


def get_config_file(ctx):
    """Get the config file option path from context for either commands or groups.

    Lets sub-groups and sub-commands access the top-level config.

    Args:
        ctx (click.Context): The Click context object.

    Returns:
        Path or None: The path to the configuration file, or None if not provided.
    """
    name = OPTION_CONFIG_FILE.name
    if ctx.obj and name in ctx.obj and ctx.obj[name]:
        return Path(ctx.obj[name])
    elif name in ctx.params and ctx.params[name]:
        return Path(ctx.params[name])
    return None


class Option(click.Option):
    """Class for storing the source type and grouping with each :class:`click.Option`.

    This class extends the :class:`click.Option` class and adds additional attributes for storing
    the data class, help group, and whether the option is a YAML reference.

    Attributes:
        data_cls (type): The Pydantic model class associated with this option.
        help_group (str): The help group text for this option.
        cli_name_path (str): The CLI name prefix for this option. Used to map to Pydantic name.
        yaml_ref (bool): Whether this option is a YAML reference option.
    """

    def __init__(self, *args, **kwargs):
        self.data_cls = kwargs.pop("data_cls", None)
        self.help_group = kwargs.pop("help_group", None)
        self.cli_name_path = kwargs.pop("cli_name_path", None)
        self.dest = kwargs.pop("dest", None)
        self.yaml_ref = kwargs.pop("yaml_ref", None)
        super(Option, self).__init__(*args, **kwargs)

    def process_value(self, ctx, value):
        value = self.type_cast_value(ctx, value)
        if ctx.obj and ctx.obj and OPTION_CONFIG_FILE.name in ctx.obj:
            return value  # HACK paramater will be validated by pydantic
        return super(Option, self).process_value(ctx, value)


class BaseCommand(click.Command):
    """A Click command that handles instantiating Pydantic models from flattened CLI params.

    This class extends the :class:`click.Command` class and overrides the :meth:`format_options` and
    :meth:`invoke` methods to handle instantiating Pydantic models from the command-line parameters.

    Methods:
        format_options(ctx, formatter): Format the command options for display.
        invoke(ctx): Invoke the command and instantiate Pydantic models from the parameters.
    """

    def format_options(self, ctx, formatter):
        format_options(self.get_params(ctx), ctx, formatter)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        #
        #  NOTE: Hack to support click.options(required=True). Set all options as NOT required if they're
        #        in the config to avoid a missing parameter error. Then restore options' required flag if they're
        #        not in the config so the CLI still handles missing parameters correctly.
        #        TODO: There must be a better way; perhaps using click.ParameterSource.FILE analogous to DEFAULT and COMMANDLINE.
        click_options = [
            (p, p.required)
            for p in self.params
            if isinstance(p, click.Option) and not issubclass(p.__class__, Option)
        ]
        for opt in click_options:
            opt[0].required = False

        cls_map = self.callback.__dict__.get("__mlsimkit_cls_map__", {})
        config_params = _make_nested_params(self.name, self.params, ctx, get_config_file(ctx), cls_map, args)

        # Restore required flag if not in config, othewise force not required to avoid missing parameter
        # during parse_args, which is safe because it's already been set by config
        for opt in click_options:
            opt[0].required = opt[1] if opt[0].name not in config_params else False

        return super().parse_args(ctx, args)

    def invoke(self, ctx):
        try:
            cls_map = self.callback.__dict__.get("__mlsimkit_cls_map__", {})
            ctx.params = _make_nested_params(self.name, self.params, ctx, get_config_file(ctx), cls_map)
            return super(BaseCommand, self).invoke(ctx)
        except Exception as e:
            if isinstance(e, click.ClickException) or ctx.obj.get("debug-mode", False):
                raise
            else:
                # avoid printing the stack
                log.error(f"Command '{self.name}' failed: {e}")
                raise click.UsageError(e, ctx)


class BaseGroup(click.Group):
    """A Click group that handles instantiating Pydantic models from flattened CLI params.

    This class extends the :class:`click.Group` class and overrides the :meth:`format_options`,
    :meth:`parse_args`, :meth:`invoke`, and :meth:`command` methods to handle instantiating
    Pydantic models from the command-line parameters.

    Methods:
        format_options(ctx, formatter): Format the group options for display.
        parse_args(ctx, args): Parse the command-line arguments and detect if help was requested.
        invoke(ctx): Invoke the group and instantiate Pydantic models from the parameters.
        command(args, kwargs): Decorate a command within the group with custom behavior.
    """

    def format_options(self, ctx, formatter):
        format_options(self.get_params(ctx), ctx, formatter)
        super().format_commands(ctx, formatter)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.obj = ctx.obj if ctx.obj else {}  # ensure obj is a dict, passed to subcommands
        # fmt: off
        # let subcommands detect if --help:
        ctx.obj["invoked_with_help"] = any((
            # when a BaseGroup is requested without args BUT let BaseCommands run without args
            args and args[-1] in self.commands and not issubclass(type(self.commands[args[-1]]), BaseCommand),
            # when --help is requested for any command
            "--help" in args,                         
            # when any parent groups have already set help=true
            ctx.obj.get("invoked_with_help", False),  
        ))

        # fmt: on
        return super().parse_args(ctx, args)

    def invoke(self, ctx):
        cls_map = self.callback.__dict__.get("__mlsimkit_cls_map__", {})
        ctx.params = _make_nested_params(self.name, self.params, ctx, get_config_file(ctx), cls_map)
        return super(BaseGroup, self).invoke(ctx)

    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], click.Command]:
        # Force sub-commands to be a mlsimkit.cli.BaseCommand class"
        if "cls" not in kwargs:
            kwargs["cls"] = BaseCommand
        elif not issubclass(kwargs["cls"], BaseCommand):
            raise Exception("Commands within a group must be an mlsimkit.cli.BaseCommand class")

        # always show defaults
        kwargs.setdefault("context_settings", {})
        kwargs["context_settings"]["show_default"] = True

        # add pass_context to ALL commands in this group
        def pass_context(command_decorator):
            def new_func(f: Callable[..., Any]) -> click.Command:
                cmd = command_decorator(f)
                cmd.callback = click.decorators.pass_context(cmd.callback)

            return new_func

        command = super(BaseGroup, self).command(*args, **kwargs)
        return pass_context(command)


class NamedEnum(Enum):
    """
    Enum that constructs members based on string names. Useful for CLI options to accept strings instead of values.

    Raises ValueError for invalid names.

    Example::

        class LevelEnum(NamedEnum):
            ERROR = 40
            INFO = 20

        LevelEnum['ERROR']
        LevelEnum['BAD_NAME'] # Raises ValueError

    CLI usage with Pydantic schemas to auto-expose names::

       class MySettings(BaseModel):
          model_config = ConfigDict(use_enum_values=True)  # special hidden field with Pydantic
          level: LevelEnum = Field(LevelEnum.ERROR.name, cli=CliExtras(use_enum_name=True))
    """

    @classmethod
    def _missing_(cls, name):
        return cls[name]


class ListParamType(click.ParamType):
    """A Click parameter that parses a delimited list of values.

    Example::

    >>> @click.command()
    >>> @click.option('--ints', type=ListParamType(int))
    >>> def cmd(ints: List[int]):
    >>>     click.echo(ints)

    >>> cmd('--ints 1,2,3')
    [1, 2, 3]
    """

    name = "list"

    def __init__(self, type: T = Any, delimiter: str = ",") -> None:
        self.type = type
        self.delimiter = delimiter
        self.type_name = type.__name__.lower() + "list"

    def get_metavar(self, param) -> str:
        return f"[{self.type_name}]"

    def convert(self, value: str, param: click.Parameter, ctx: click.Context) -> List[T]:
        try:
            if isinstance(value, list):
                return value  # support CLIRunner,invoke(...) for testing (TODO: why is this necessary?)
            elif get_origin(self.type) == Literal:
                return [x.strip() for x in value.split(self.delimiter)]
            else:
                return [self.type(x.strip()) for x in value.split(self.delimiter)]
        except ValueError:
            self.fail(f"{value!r} is not a valid {self.type_name}")


class ShorthandParamType(click.ParamType):
    """
    Click Parameter type for a Pydantic model that accepts shorthand notation.

    This parameter type allows users to specify values for a Pydantic model using a
    shorthand notation in the command line. It also supports loading values directly
    from a configuration file.

    Usage example:

    .. code-block:: python

        from pydantic import BaseModel
        from typing import List

        class MyOption(BaseModel):
            name: str
            columns: List[str]

        @click.command()
        @click.option(
            "-o", "--option",
            type=ShorthandParamType(MyOption),
            help="Specify an option in shorthand notation: 'name=value,columns=col1 col2'"
            callback=lambda ctx, param, value: value)
        def my_command(option: MyOption):
            print(option)

        if __name__ == '__main__':
            my_command()

    From the command line:

    .. code-block:: bash

        $ python script.py -o "name=my_option,columns=col1 col2"
        name='my_option' columns=['col1', 'col2']

    Alternatively, values can be loaded from a configuration file when using  :func:`program`:


    .. code-block:: yaml

        # config.yaml
        option:
          name: my_option
          columns:
            - col1
            - col2

    .. code-block:: bash

        $ python script.py --config config.yaml

    Args:
        model (Type[BaseModel]): The Pydantic model class to create an instance from
            the shorthand string or configuration file.
    """

    def __init__(self, model, delimiter=",", quotechar="|"):
        self.model = model
        self.name = model.__name__  # required for --help
        self.delimiter = delimiter
        self.quotechar = quotechar

    def convert(self, value, param, ctx):
        """
        Convert to an instance of the Pydantic model.

        This method handles converting different types of values to an instance of the
        Pydantic model associated with this parameter type. It supports three main cases:

        1. If the value is a list, it assumes that the parameter is marked as `multiple=True`.
           In this case, it recursively converts each element of the list to the Pydantic
           model and returns a list of model instances to match Click built-in behavior.
           If the parameter is not marked as `multiple=True`, it raises a `BadParameter` exception.

        2. If the value is a dictionary, it assumes that the value is loaded directly from
           a configuration file. In this case, it creates an instance of the Pydantic model
           by passing the dictionary as keyword arguments. This enables using config-only
           even for ShorthandParamTypes options.

        3. If the value is a string, it assumes that the string represents a shorthand
           notation for the Pydantic model. It calls the `_parse_param_shorthand_str`
           function to parse the string and create an instance of the Pydantic model.

        If the value is not a list, dictionary, or string, it raises a `BadParameter`
        exception.

        Args:
            value (Any): The value to be converted to a Pydantic model instance.
            param (click.Parameter): The Click parameter object associated with the value.
            ctx (click.Context): The Click context object associated with the value.

        Returns:
            BaseModel: An instance of the Pydantic model associated with this parameter type.

        Raises:
            click.BadParameter: If the value is a list and the parameter is not marked as
                `multiple=True`, or if the value is not a list, dictionary, or string.
        """
        if isinstance(value, list):
            if param.multiple:
                return [self.convert(v, param, ctx) for v in value]
            else:
                raise click.BadParameter(
                    f"Parameter has multiple=False but received a list: '{value}'", param=param
                )
        if isinstance(value, dict):
            # support loading directly from config
            return self.model(**value)
        if isinstance(value, str):
            return _parse_param_shorthand_str(value, self.model, self.delimiter, self.quotechar)
        else:
            raise click.BadParameter(f"Unexpected value type '{type(value)}', value='{value}'", param=param)


class ResourcePath(click.Path):
    """A Click option type that resolves paths from the provided search paths,
    including paths from importlib.resources.

    This class inherits from `click.Path` and extends its functionality to support
    resolving paths from package resources or folder paths. If the provided value is not a valid
    existing regular file path, it attempts to resolve the path using the specified
    search paths, which are package paths relative to importlib.resources.

    If require=true and the file does not exist after all search paths then a
    BadParameter is raised.

    Parameters
    ----------
    search_paths: Optional[Sequence[str]]
        A sequence of package paths to search for the provided value. If not provided,
        the class will only attempt to resolve regular file paths.

    Attributes
    ----------
    search_paths: Sequence[str]
        The sequence of package paths to search for the provided value.

    Methods
    -------
    convert(value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Union[str, None]
        Converts the provided value to a valid file path.

        If the value is a valid existing regular file path, it returns the resolved path.
        If the value is not a valid regular file path, it attempts to resolve the path
        using the specified search paths. If the path is found in one of the search paths
        and the resolved path exists, it returns the resolved path.

        If the value cannot be resolved as a valid existing file path, and the option
        is required, it raises a `click.exceptions.BadParameter` exception. If the option
        is not required, it returns `None`.

        Parameters
        ----------
        value: str
            The value to convert to a file path.
        param: Optional[click.Parameter]
            The Click parameter object associated with the value.
        ctx: Optional[click.Context]
            The Click context object associated with the value.

        Returns
        -------
        Union[str, None]
            The resolved file path if the value is a valid existing file path or can be
            resolved using the search paths. `None` if the value cannot be resolved and
            the option is not required.
    """

    def __init__(self, search_paths=None, *args, **kwargs):
        self.search_paths = search_paths or []
        super().__init__(*args, **kwargs)

    def convert(self, value, param, ctx):
        # First, try to interpret the value as a regular file path
        try:
            path = super().convert(value, param, ctx)
            if os.path.exists(path):
                return path
        except click.exceptions.BadParameter:
            pass

        # If the value is not a valid regular file path, try to resolve it using the search paths
        for search_path in self.search_paths:
            try:
                # python 3.9 needs modules for the full path so we split
                root = importlib.resources.files(search_path.split(".")[0])
                if root:
                    filepath = Path(*search_path.split(".")[1:]) / Path(value)
                    return str(root / filepath)
            except ImportError:
                pass

        # If the value is not found in any of the search paths, raise an error if required
        if param.required:
            raise click.exceptions.BadParameter(f"Could not resolve required path '{value}'")
        return None

    def get_metavar(self, param):
        return "PATH"


class PydanticOptions:
    """Utility to generate Click options from Pydantic models.

    Usage:
    >>> class MySettings(pydantic.BaseModel):
           level: int
           mode: bool

    >>> @mlsimkit.cli.command()
    >>> @mlsimkit.cli.options(MySettings, dest="settings")
    >>> def cmd(settings: MySettings):
    >>>     click.echo(settings.dict())

    Alternatively, without decorators:

    >>> options = PydanticOptions.from_model(
           MySettings,
           dest="settings",
           prefix="my",
           help_group="My Settings")

    >>> print([o.name for o in options])
         # ['--my-level', '--my-mode/--my-no-mode]

    Description:
      This class handles introspecting Pydantic models and generating
      the corresponding Click options.

      Handles translating Pydantic fields to appropriate Click options:
       - Bool field becomes flag (--flag/--no-flag)
       - Enum fields become click.Choice
       - Optional fields remain optional
       - Help mesage is derived from the Field(..., description="...") string
       - Default is derived from the Field default value.

      NOTE: Union types raise click.BadParameter

    Arguments:
        model: The Pydantic model class
        dest: The attribute to store parsed data on CLI context
        prefix: Optional prefix added to option names
        help_group: The help text section name for options

    """

    @staticmethod
    def get_name(opt_type: type, field_name: str, name_prefix: str) -> str:
        """Convert field name to option name."""
        name = field_name.replace("_", "-")  # TODO: detect delimiter  conflicts
        if opt_type is bool:
            # automatically make a flag with '/' separator
            return (
                f"--{name_prefix}{DELIMITER}{name}/--{name_prefix}{DELIMITER}no-{name}"
                if name_prefix
                else f"--{name}/--no-{name}"
            )
        else:
            return f"--{name_prefix}{DELIMITER}{name}" if name_prefix else f"--{name}"

    @staticmethod
    def get_cli_extras(field) -> CliExtras:
        """Get CLI extras metadata for field."""
        return field.json_schema_extra.get("cli", None) if field.json_schema_extra else CliExtras()

    def get_option_type(field_name: str, field, model: Type[BaseModel]) -> type:
        """Get allowed option type from field.

        Handles converting Pydantic Optional to required option type. Rejects all
        other Union types.

        Note: Pydantic supports Union[] fields but Click does not. Instead, we
              convert Pydantic Optional[] fields and reject all other Unions.

        Args:
            field_name: Name of the model field
            field: Pydantic model field
            model: Pydantic model

        Returns:
            Allowed option type for Click

        Raises:
            click.BadParameter: If unsupported Union type
        """
        opt_type = field.annotation
        if get_origin(opt_type) is not Union:
            return opt_type

        types = get_args(field.annotation)
        is_optional_type = len(types) == 2 and types[1] is type(None)
        if is_optional_type:
            return types[0]

        raise click.BadParameter(
            f"Union not supported: {model.__module__}.{model.__name__}.{field_name}: {opt_type}"
        )

    @staticmethod
    def is_nested_field(field_annotation: type, BaseModel: Type[BaseModel]) -> bool:
        """Check if field is a nested Pydantic model."""
        return lenient_issubclass(field_annotation, BaseModel)

    @staticmethod
    def handle_nested_option(
        field_name: str,
        field: FieldInfo,
        dest: str,
        name_prefix: str,
        help_group: str,
        yaml_file: Union[bool, Dict] = False,
        delimiter: str = "__",
    ) -> List[Option]:
        """Recursively handle nested model option."""
        cli = field.json_schema_extra.get("cli", None) if field.json_schema_extra else CliExtras()
        name_prefix = name_prefix + DELIMITER if name_prefix else ""  # if parent has no cli prefix
        prefix = cli.prefix or field_name or None
        title = (
            help_group
            if help_group
            else cli.title or (field.annotation.Config.title if hasattr(field.annotation, "Config") else None)
        )

        # nested options are added recursively
        return PydanticOptions._yield_options(
            field.annotation,
            dest=f"{dest}{delimiter}{field_name}",
            name_prefix=f"{name_prefix}{prefix}",
            help_group=title,
            yaml_file=yaml_file,
            delimiter=delimiter,
        )

    @staticmethod
    def _yield_options(
        model: ModelMetaclass,
        dest: str,
        name_prefix: str = None,
        help_group: str = None,
        yaml_file: Union[bool, Dict] = False,
        delimiter: str = "__",
    ) -> Iterator[Option]:
        # Add a hidden option for each BaseModel type to allow overriding the nested field from the CLI
        #  For simple usage, yaml_file can be True or customized with a dictionary of values
        if (isinstance(yaml_file, bool) and yaml_file) or (
            isinstance(yaml_file, dict) and yaml_file is not None
        ):
            yaml_file = yaml_file if isinstance(yaml_file, dict) else {}
            opt_name = "--" + yaml_file.get("name", f"{name_prefix}" if name_prefix else f"{dest}")
            log.warning(
                "Warning: `yaml_file` flag is experimental, found for schema '%s' (option '%s')",
                model.__name__,
                opt_name,
            )
            yield Option(
                param_decls=[opt_name, dest],
                data_cls=model,
                hidden=yaml_file.get("hidden", True),
                type=click.Path(exists=False),  # hack, allow '+' relative files and check exists later),
                yaml_ref=True,
                help=yaml_file.get("help", ""),
                help_group=help_group,
            )

            #  We only support first-level references, for simplicity. Clear yaml_file means nested
            #  BaseModel types are ignored when recursing.
            yaml_file = None

        for field_name, field in model.model_fields.items():
            if PydanticOptions.is_nested_field(field.annotation, BaseModel):
                yield from PydanticOptions.handle_nested_option(
                    field_name,
                    field,
                    dest,
                    name_prefix,
                    help_group,
                    yaml_file,
                )
                continue

            cli = PydanticOptions.get_cli_extras(field)
            if cli and cli.exclude:
                continue

            opt_type = PydanticOptions.get_option_type(field_name, field, model)
            name = PydanticOptions.get_name(opt_type, field_name, name_prefix)
            cli_name_path = [name_prefix, field_name] if name_prefix else [field_name]
            default = field.default if field.default is not PydanticUndefined else None
            help_msg = field.description or ""
            multiple = False

            if isclass(opt_type) and issubclass(opt_type, Enum):
                if default is not None and isinstance(default, Enum):
                    default = default.value # work-around Click 8.2.0 breaking change
                opt_type = click.Choice([e.name if cli.use_enum_name else e.value for e in opt_type])
            elif issubclass(_GenericAlias, type(opt_type)):
                if issubclass(list, get_origin(opt_type)):
                    opt_type = ListParamType(type=get_args(opt_type)[0], delimiter=",")
                else:
                    raise click.BadParameter(
                        f"Unsupported parameter type: {model.__module__}.{model.__name__}.{field_name}: {opt_type}"
                    )

            yield Option(
                param_decls=[name, f"{dest}{delimiter}{field_name}"],
                type=opt_type,
                default=default,
                required=field.is_required(),
                multiple=multiple,
                hidden=cli.hidden if cli else False,
                data_cls=model,
                cli_name_path=cli_name_path,
                dest=dest,
                help_group=help_group,
                help=help_msg,
            )

    @staticmethod
    def from_model(
        model: Type[BaseModel],
        dest: str,
        name_prefix: str = None,
        help_group: str = None,
        yaml_file: Union[bool, Dict] = False,
    ) -> List[Option]:
        """Generate list of Click options from Pydantic model.

        Args:
            model: The Pydantic model class to generate options for
            dest: The destination attribute to store parsed data
            name_prefix: Optional prefix to add to option names
                 to prevent duplicate name conflicts
            help_group: Help text to group options in CLI

        Returns:
            List of Click Option instances

        The model argument specifies the Pydantic schema model to introspect.

        The dest argument controls the attribute name that the parsed data
        will be mapped to in the CLI context.

        The name_prefix allows prepending each option to prevent duplicate
        name conflicts when multiple models are used.

        The help_group text sets the named section for options when displayed
        in the CLI help text.
        """
        return list(PydanticOptions._yield_options(model, dest, name_prefix, help_group, yaml_file))


def _cls_map_from_params(params: List[Option], delimiter: str = "__") -> Dict[str, type]:
    """Construct map of param destination names to Pydantic model classes.

    This maps the first part of each param name to its Pydantic model class.
    Keys without a delimiter are ignored and considered as values, not a Pydantic
    model class.

    Args:
        params: List of Option mapping Click.option to Pydantic models
                with a `data_cls` attribute

    Returns:
        Dict mapping param dest name to its model class

    Examples:
        params = [
            Model(name='modela__foo', data_cls=ModelA),
            Model(name='modelb__bar', data_cls=ModelB)
        ]
        cls_map = _cls_map_from_params(params)
        print(cls_map)

        # {'modela': ModelA, 'modelb': ModelB}
    """
    cls_map = {}
    for p in params:
        if hasattr(p, "data_cls"):
            v = cls_map.get(p.name, p.data_cls)
            if v != p.data_cls:
                raise Exception(
                    f"Same parameter names '{p.name}' map to different classes ({v} != {p.data_cls})"
                )
            cls_map[p.name] = v
    return cls_map


def format_options(params, ctx, formatter):
    # from https://github.com/pallets/click/issues/373
    opts = OrderedDict()
    for param in params:
        rv = param.get_help_record(ctx)
        if rv is not None:
            if hasattr(param, "help_group") and param.help_group:
                opts.setdefault(str(param.help_group), []).append(rv)
            else:
                opts.setdefault("Options", []).append(rv)

    for name, opts_group in opts.items():
        with formatter.section(name):
            formatter.write_dl(opts_group)


def _make_nested_params(cmd_name, param_types, ctx, root_config_file, cls_map, args=[]):
    """
    Create a nested dictionary of parameter values from command-line arguments, config files, and default values.

    This function processes the parameters for a Click command, combining values from various sources
    (command-line arguments, config files, and default values) into a nested dictionary structure.
    The resulting dictionary can then be used to instantiate Pydantic models with the parsed parameter values.

    Args:
        cmd_name (str): The name of the Click command being processed.
        param_types (List[Option]): A list of Click options representing the command parameters.
        ctx (click.Context): The Click context object, containing information about the current command invocation.
        root_config_file(Union[str, Path, None]): The path to the configuration file, if provided.

    Returns:
        Dict[str, Any]: A nested dictionary containing the processed parameter values.

    Raises:
        click.BadParameter: If there are validation errors when parsing the parameter values.

    Notes:
        - The function supports nested parameter structures by using a delimiter ('__') in parameter names.
        - It merges values from multiple sources in the following order:
          1. Default values from the base configuration file
          2. Values from additional YAML files specified with '+' references
          3. User-specified command-line arguments
        - YAML files referenced with '+' are resolved relative to the base configuration file's directory.
        - If a parameter is specified both in a YAML file and on the command line, the command-line value takes precedence.
        - The resulting dictionary keys correspond to the Pydantic model classes specified for each parameter.
    """

    log.debug("Processing nested parameters for command '%s'", cmd_name)

    #
    # 1. Load the root/base config file
    #
    root_config = {}
    root_config_file = Path(root_config_file) if root_config_file else None
    if root_config_file:
        try:
            with open(root_config_file, encoding="utf-8") as f:
                # uses safe loader via UniqueKeyLoader
                root_config = yaml.load(f, Loader=conf.UniqueKeyLoader)  # nosec B506
        except BaseException as e:
            raise click.UsageError(f"Failed loading {root_config_file.as_posix()!r}: {e}")

    log.debug("Loaded root config for command '%s', root_config=%s", cmd_name, root_config)

    #
    # 2. Load "defaults" from the base config
    #
    #   "defaults" is a special key for configs that triggers merging across files based on the "+" references.
    defaults, _ = conf.get_nested_value(root_config, ["defaults"] + get_command_path(ctx)[1:], {})
    config, _ = conf.get_nested_value(root_config, get_command_path(ctx)[1:], {})

    config = conf.merge_configs(
        defaults,
        config,
        value_filter=conf.is_nested_ref,  #  nosemgrep: is-function-without-parentheses
        callback=lambda value, keys: conf.load_yaml(root_config_file.parent, value, keys[2:]),
        # note ^^: remove "command.defaults" from keys to get the actual pydantic object in defaults
    )

    log.debug("Defaults loaded, config=%s", json.dumps(config, indent=2))

    #
    # 3. Merge YAML file reference options OVER the default config using command-line option order first.
    #
    yaml_ref_options = {p.name: p for p in param_types if isinstance(p, Option) and p.yaml_ref}
    config = conf.merge_yaml_refs(config, root_config_file, yaml_ref_options, ctx.params)

    log.debug("YAML file references merged, config=%s", json.dumps(config, indent=2))

    #
    # 4. Convert naming from CLI options format to Pydantic-ready format
    #       (Replaces '-' with '_' and prefixes the nested field names to the config.)
    key_mapping = {}
    for p in param_types:
        if "cli_name_path" in p.__dict__ and p.cli_name_path:
            old_key = sum([s.split("-") for s in p.cli_name_path], [])
            new_key = (p.dest.split("__") if p.dest else p.name.split("__")) + [p.cli_name_path[-1]]
            key_mapping[".".join(old_key)] = ".".join(new_key)

    log.debug("Converting config names, key mapping=%s", key_mapping)
    config = conf.replace_chars_in_keys(config, "-", "_")
    config = conf.move_keys(config, key_mapping)

    log.debug("Config naming converted, config=%s", json.dumps(config, indent=2))

    #
    # 5. Populate click.Options (but not Pydantic Options) from config, allows click.Options in config only
    # fmt: off
    opts = [
        p 
        for p in param_types
        if isinstance(p, click.Option) and not issubclass(p.__class__, Option) and p.name in config \
        and ctx.get_parameter_source(p.name) != ParameterSource.COMMANDLINE # only set when user has NOT specified
    ]
    # fmt: on

    for p in opts:
        ctx.params[p.name] = p.type.convert(config.get(p.name), p, ctx)

    log.debug("Command click.options populated from config, ctx.params=%s", ctx.params)

    #
    # 6. Merge user-specified command-line options OVER .yaml sources (in the order specified)
    #       NOTE: This means yaml file sources are overridden even if the yaml file
    #       argument is specified after the 'normal' command-line option. We can change this
    #       but the logic here maintains consistency where all file inputs are overrridden by
    #       cmdline values.
    #
    for k, v in ctx.params.items():
        user_specified = ctx.get_parameter_source(k) != ParameterSource.DEFAULT
        yaml_source = k in yaml_ref_options
        if user_specified and not yaml_source:
            conf.set_nested_value(data=config, keys=k.split("__"), value=v)

    log.debug("User-specified cmdline options applied over config, ctx.params=%s", ctx.params)

    #
    # 7. Check if any required options are missing
    #
    for p in param_types:
        _, loaded_from_config = conf.get_nested_value(config, p.name.split("__"))
        on_the_cmdline = bool(set(p.opts) & set(args))
        if p.required and not (
            loaded_from_config or on_the_cmdline or p.value_is_missing(ctx.params.get(p.name, None))
        ):
            log.debug("Required parameter '%s' is missing, even after loading config: %s", p, config)
            raise click.MissingParameter(ctx=ctx, param=p)

    #
    # 8. Finally, convert parameters from dict to the Pydantic classes
    #
    try:
        cls_map = {**cls_map, **_cls_map_from_params(param_types)}
        nested_params = conf._make_nested_dict(
            {k: v for k, v in ctx.params.items() if k not in yaml_ref_options}
        )
        return {
            k: cls_map[k](**{**v, **config.get(k, {})}) if k in cls_map else v
            for k, v in nested_params.items()
        }
    except ValidationError as e:
        raise click.BadParameter(e)


def _check_duplicate_option_names(params: List[click.Option]) -> None:
    """Check for duplicate option names across click.Options

    Args:
        params: List of Click options

    Raises:
        click.BadParameter: If duplicate names found

    Examples:
        params = [
            click.Option(names=['--modela']),
            click.Option(names=['--modelb'])
        ]
        _check_duplicate_option_names(params) # passes

        params = [
            click.Option(names=['--model']),
            click.Option(names=['--model'])
        ]
        _check_duplicate_option_names(params) # Raises exception
    """

    names: Set[str] = set()
    for param in params:
        duplicate_names = names & set(param.opts)
        if duplicate_names:
            raise click.BadParameter(f"Duplicate option names, use a prefix: {duplicate_names}")
        names |= set(param.opts)


def _parse_param_shorthand_str(
    param_str: str, model_cls: Type[BaseModel], delimiter=",", quotechar="|"
) -> BaseModel:
    """
    Parse a string of comma-separated key-value pairs into a Pydantic model instance.

    This function takes a string representing parameter key-value pairs and a Pydantic
    BaseModel class. It parses the string, extracting the keys and values, and creates an
    instance of the provided BaseModel class with the parsed values. The function handles
    both singular values and list values based on the field types defined in the BaseModel.

    Args:
        param_str (str): A string containing comma-separated key-value pairs in the format
            "key1=value1, key2=value2, ..., keyN=valueN". List values are space-separated,
            e.g., "key=value1 value2 value3".
        model_cls (Type[BaseModel]): A Pydantic BaseModel class to instantiate with the
            parsed values.

    Returns:
        BaseModel: An instance of the provided BaseModel class with the parsed values.

    Raises:
        ValueError: If the parameter format is invalid or if a duplicate key is found.
        ValidationError: If the parsed values do not conform to the field types defined
            in the BaseModel class.

    Example:
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     ids: List[int]
        ...
        >>> param_str = "name=John Doe, ids=1 2 3"
        >>> model_instance = _parse_param_shorthand_str(param_str, MyModel)
        >>> print(model_instance)
        name='John Doe' ids=[1, 2, 3]
    """
    param_dict: Dict[str, Any] = {}

    params = list(csv.reader(StringIO(param_str), delimiter=delimiter, quotechar=quotechar))[0]
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            key = key.strip()
            field = model_cls.model_fields.get(key, None)
            if not field:
                raise click.BadParameter(f"Unknown key '{key}'")
            field_type = PydanticOptions.get_option_type(key, field, model_cls)

            # handle List[...] field types
            value_type = str  # non-list value types are automatically converted from string by pydantic
            if issubclass(_GenericAlias, type(field_type)):
                if issubclass(list, get_origin(field_type)):
                    value_type = get_args(field_type)[0]
                    field_type = list
                else:
                    raise click.BadParameter(
                        f"Unsupported parameter type: {model_cls.__module__}.{model_cls.__name__}.{key}: {field_type}"
                    )

            if issubclass(list, field_type):
                value = [value_type(v.strip()) for v in value.split(" ") if v.strip()]  # non-empty values

            if key in param_dict:
                raise click.UsageError(f"Duplicate key '{key}'")

            param_dict[key] = value

    return model_cls(**param_dict)
