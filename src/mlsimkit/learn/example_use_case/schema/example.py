# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from mlsimkit.common.schema.cli import CliExtras
from mlsimkit.common.cli import NamedEnum

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class Settings(BaseModel):
    property_one: str = "DefaultValue1"
    level: int = Field(0, ge=0, lt=10)
    values: List[float] = Field([2, 0, 1])

    # Uncomment the following field to have a required field
    # this_property_is_required: str


#
# Example using more advanced settings:
#  - nested models
#  - using enums names
#  - hide and exclude fields on the CLI
#  - unknown fields are forbidden with extra="forbid"
#


# Enum names can be exposed on the CLI as strings for user-friendly interfaces. The enum can
# map to any value type. For example, below we map names to floats but the CLI is P50|P95|P99.
class PercentileEnum(NamedEnum):
    P50 = 0.5
    P95 = 0.95
    P99 = 0.99


# Nested models are supported, and benefit from using a CLI naming prefix.
class NestedSettings(BaseModel, extra="forbid"):
    the_property: str = Field("NestedProperty")
    # Fields can be hidden from help in the CLI. Hidden fields still work if specified.
    hidden_property: str = Field("NestedHiddenProperty", cli=CliExtras(hidden=True))
    # Fields can be excluded from the CLI. They are still configurable from YAML.
    excluded_property: Optional[str] = Field("NestedExcludedProperty", cli=CliExtras(exclude=True))


class MoreSettings(BaseModel):
    property_one: str = Field("Brg", description="Field with duplicate name to ExampleSettings schema")
    property_two: Optional[str] = Field(None, description="Optional string field", max_length=10)
    nested_settings: NestedSettings = Field(NestedSettings(), cli=CliExtras(prefix="nested"))

    # NamedEnum types use strings for input values, so either "P50" or the .name member:
    percentile: PercentileEnum = Field(PercentileEnum.P50.name, cli=CliExtras(use_enum_name=True))

    # "model_config" is an alternative to configure a BaseModel. It is special field unseen by users.
    # See https://docs.pydantic.dev/latest/concepts/config/
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
