# Preparing the later GitHub push

No push is performed as part of this preparation. Run these steps only when you are ready and have decided whether this is a separate repository or a folder in the group repository.

## What belongs in Git

Source, tests, configuration, notebook, README, guides, small verified reports, citation metadata and the GitHub test workflow. Data, environments, checkpoints, caches and source Word files are ignored. This project does not choose an open-source license on behalf of the whole team; agree one before public reuse permissions are granted.

## Separate repository

From the `Rohit_DL_Project` folder, inspect whether it is already inside a repository. If starting a separate repository, use:

```bash
git init
git add .
git status --short
git diff --cached --stat
python -m pytest -q
git commit -m "Add TextCNN sentiment and robustness pipeline"
git branch -M main
```

Create/select the intended GitHub repository, then add its real URL with `git remote add origin YOUR_REPOSITORY_URL` and push with `git push -u origin main`. Replace the placeholder; never use a guessed URL. Check `git status` before committing so raw datasets and local model files are not staged.

## Group repository

Keep this package in an agreed subdirectory or merge it into the team's structure. The included workflow assumes this project is at the repository root; if nested, set the run working directory and pip cache dependency path to this folder, or integrate its test command into the team's existing workflow. Coordinate package names and shared data locations before merging.

## Small evidence versus large files

The checked-in report summaries should identify actual runs and include provenance, counts and metric definitions. Store large checkpoints in a release attachment or agreed artifact store with their vocabulary/configuration. Do not commit downloaded review text or entire `runs/` directories accidentally.
