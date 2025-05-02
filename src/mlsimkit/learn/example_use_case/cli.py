# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import logging

import mlsimkit

from .schema.example import Settings, MoreSettings, PercentileEnum

log = logging.getLogger(__name__)

"""
    Usage:

        $ mlsimkit-learn  example-use-case --help
        [MLSimKit] Learning Tools
        Usage: mlsimkit-learn example-use-case [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                               [ARGS]...]...

        Options:
          --common-option TEXT
          --help                Show this message and exit.

        Commands:
          preprocess
          test
          train

   Example 1. Using the 'test' command with nested arguments:

       $ mlsimkit-learn --log-level info example-use-case --common-option "foo"  test

   Example 2. Chaining multiple commands (test > preprocess > train): 

       $ mlsimkit-learn example-use-case 
           test --property-one "foo" \
           preprocess --property-one "bar" \
           train --opt-type-name "adamw"


   Example 3. Use a shared YAML config file: 

       $ mlsimkit-learn  --config myconfig.yaml example-use-case test

        myconfig.yaml:
          logging:
            level: error

          test: 
            settings:
              property_one: "MyPropertyFromConfig"
              level: 5

            more_settings:
              property_one: "AnotherPropertyFromConfig"


    Example 4: Use YAML file to set a schema-group of options:

       $ mlsimkit-learn  --config myconfig.yaml example-use-case test --settings mysettings.yaml

        mysettings.yaml:
          property_one: "PropertyFromSettingsYAML"
          level: 9
          values: [9,8,7,6,5,4,3,2,1]


    Example 5: Override YAML configurations with the command-line:

       $ mlsimkit-learn  --config myconfig.yaml example-use-case test --settings mysettings.yaml --level 0
       ...
        [INFO] Running command 'test'
        [INFO] 	settings: {'property_one': 'DefaultValue1', 'level': 0, 'values': [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]}
"""


#
# Multiple commands can be grouped and then share a context.
#  - 'common-option' is available to all subcommands via 'ctx.obj'
#
@mlsimkit.cli.group(chain=True)
@click.option("--common-option", type=str, default="Default")
def example_use_case(ctx: click.Context, common_option: str):
    """
    Demonstrate CLI framework for developers
    """
    log.info("Use Case: Example use case to demonstrate CLI framework")
    log.debug("A debug message")
    ctx.obj["common_option"] = common_option


#
# A command is invoked after the parent groups.
# Pydantic schemas can be used as options to avoid enumerating
# each field individually. The fields are automatically exposed as click
# options.
#
# Below, the Setting schema is exposed and given to the test()
# function in the 'settings' parameter. Each field in Settings
# is a CLI option. 'yaml_file=True' adds an option that accepts
# a path to a YAML file ('--settings PATH')
#
@example_use_case.command()
@mlsimkit.cli.options(Settings, dest="settings", yaml_file=True)
def test(ctx: click.Context, settings: Settings):
    log = logging.getLogger(f"{__name__}.test")
    log.info("Running command 'test'")
    log.info("\tsettings: %s", settings.dict())


#
# There are more advanced options:
#  - Schemas can accept YAML file as inputs by using yaml_file=True or yaml_file={...}.
#   - 'hidden': when true, the yaml file option is hidden from --help
#   - 'name': is the name of the yaml file option. This is optional and defaults to 'prefix' if present otherwise to the 'dest' naming.
#  - Multiple schemas can be used as options:
#   - 'prefix' parameter prefixes the name to the CLI options. This is useful to avoid duplicate field names.
#   - 'help_group' formats the options for this schema into a group upon --help
#  - Ordinary click.option() can be used alongside Pydantic schema options.
#    - 'percentile' is a NamedEnum for exposing strings as inputs on the CLI instead of enum values.
#
@example_use_case.command()
@mlsimkit.cli.options(Settings, dest="settings", yaml_file={"hidden": False, "name": "settings-file"})
@mlsimkit.cli.options(
    MoreSettings, dest="groupB", prefix="groupB", help_group="Additional Settings", yaml_file=True
)
@click.option("--percentile", type=PercentileEnum, default=PercentileEnum.P50.name)
def preprocess(ctx: click.Context, settings: Settings, groupB: MoreSettings, percentile: PercentileEnum):
    log = logging.getLogger(f"{__name__}.preprocess")
    log.info("Running command 'preprocess'")
    log.debug("\tctx.obj: %s", ctx.obj)
    log.info("\tsettings: %s", settings.dict())
    log.info("\tgroupB: %s", groupB.dict())
    log.info("\tpercentile: %s=%s", percentile.name, percentile.value)

    # lazy load to only import when this command is invoked
    from .preprocessing import run_preprocess

    run_preprocess(settings, groupB)


#
#
#
@example_use_case.command()
@mlsimkit.cli.options(Settings, dest="settings")
@click.option("--another-option", type=str, default="DefaultValue")
def train(ctx: click.Context, settings: Settings, another_option: str):
    log = logging.getLogger(f"{__name__}.train")
    log.info("Running command 'train'")
    log.debug("\tctx.obj: %s", ctx.obj)
    log.debug("\tsettings: %s", settings.dict())
    log.debug("\tanother_option: %s", another_option)

    # lazy load to only import when this command is invoked
    from .training import run_train

    run_train(settings, another_option)
