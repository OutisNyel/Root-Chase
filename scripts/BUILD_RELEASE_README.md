# GorgeChase Release Packaging

Build a release directory or zip with only the files needed by the container
runtime:

```text
code/
  conf/
  agent_ppo/
  .vscode/
  kaiwu.json
  train_test.py
```

The scripts are public-release safe by default. They exclude local caches,
logs, TensorBoard outputs, and model checkpoint artifacts such as
`agent_ppo/ckpt/` and `*.pkl`.

## Python

```bash
python scripts/build_release.py
python scripts/build_release.py --zip
python scripts/build_release.py --output ./releases --name my-release
```

Include checkpoint files only when you intentionally need a private runtime
package with a preload model:

```bash
python scripts/build_release.py --include-checkpoints
python scripts/build_release.py --zip --include-checkpoints
```

## PowerShell

```powershell
.\scripts\build_release.ps1
.\scripts\build_release.ps1 -UseZip
.\scripts\build_release.ps1 -OutputDir ".\releases" -ArchiveName "my-release"
```

Include checkpoint files only for a private runtime package:

```powershell
.\scripts\build_release.ps1 -IncludeCheckpoints
.\scripts\build_release.ps1 -UseZip -IncludeCheckpoints
```

## Notes

- Default output is `release/` under the repository root.
- The latest directory or zip is refreshed after each successful build.
- Public releases should not include `doc/html/`, `doc/md/`, `train_doc/`,
  `train_monitor/tensorboard/`, or checkpoint files.
