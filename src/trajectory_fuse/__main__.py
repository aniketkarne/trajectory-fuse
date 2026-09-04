"""Allow ``python -m trajectory_fuse ...`` to invoke the CLI.

The installed wheel registers a ``trajectory-fuse`` console script (see
``[project.scripts]`` in pyproject.toml). This module is the equivalent
for users who prefer the module form — useful in environments where the
entry-point script is shadowed, and required for ``python -m trajectory_fuse``
in test harnesses that shell out to the module.
"""

from trajectory_fuse.cli import main

raise SystemExit(main())