# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

from accelerate import PartialState

from mlsimkit.common.logging import getLogger

log = getLogger(__name__)


class BaseProjectContext(BaseModel):
    """
    Data class for persisting state to disk between mlsimkit-learn commands.
    Used to chain commands. Safe for multi-processing via 'accelerate launch`.
    """

    outdir: Optional[str] = None
    run_id: Optional[str] = None

    @classmethod
    def init(cls, ctx, output_dir):
        ctx.obj["output_dir"] = str(output_dir)

    @classmethod
    def get(cls, ctx):
        return ctx.obj[".project"]

    @classmethod
    def load(cls, ctx):
        log.debug("Loading project context")
        if ctx.obj["invoked_with_help"]:
            return cls()

        if "output_dir" not in ctx.obj:
            raise RuntimeError(
                "Expected 'output_dir' in context, this command should be a 'mlsimkit-learn' sub-command"
            )

        outdir = Path(ctx.obj["output_dir"])
        ctx.obj[".project-file"] = outdir / ".project"
        ctx.obj[".project"] = cls()
        path = ctx.obj[".project-file"]
        if path.exists():
            log.debug("Loading .project file '%s'", path)
            with path.open("r") as f:
                s = yaml.safe_load(f)
                ctx.obj[".project"] = cls(**s if s else {})

        ctx.obj[".project"].outdir = str(outdir.resolve())
        return ctx.obj[".project"]

    def save(self, ctx, exist_ok=True):
        if ctx.obj["invoked_with_help"]:
            return

        with_accelerate = ctx.obj.get("invoked_with_accelerate", False)
        if with_accelerate and not PartialState().is_main_process:
            # in multi-process mode, so wait until main process writes the .project file
            PartialState().wait_for_everyone()
            return

        path = ctx.obj[".project-file"]

        # 'x' mode ensures callers create the file, otherwise get FileExistsError
        open_mode = "w" if exist_ok else "x"
        with path.open(open_mode) as f:
            log.debug("Writing .project file '%s'", path)
            f.write(yaml.dump(self.dict()))

        if with_accelerate and PartialState().is_main_process:
            # signal to other processes main process is done
            PartialState().wait_for_everyone()
            return
