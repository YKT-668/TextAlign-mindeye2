# Contributing

Open an issue before large changes. Keep historical `textalign` module and
checkpoint field names when compatibility depends on them. New code should:

- avoid machine-specific paths and credentials;
- keep data and checkpoints outside Git;
- include a CPU-sized test where practical;
- document changes to Final14 protocols or artifact revisions;
- preserve upstream notices and licenses.

Run `python -m unittest discover -s tests/smoke -v` and the validation commands
in `docs/reproduction.md` before submitting a change.
