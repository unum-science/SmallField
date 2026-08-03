# Git Hooks

`pre-commit` and `commit-msg` are the rulebook — each check carries its own rationale as a comment directly above it, and the header of `pre-commit` records the three rules that matter but cannot be matched by a regex.

## Activation

Configuring the CMake build arms them:

```sh
cmake -B build .
git config core.hooksPath   # should print .githooks
```

Arming is skipped when SmallField is built as a subproject, because mutating a consumer's Git config is not this project's business, and when there is no `.git` directory, as in a tarball build.
To arm them without configuring a build:

```sh
git config core.hooksPath .githooks
```

Note that `core.hooksPath` and `pre-commit install` are mutually exclusive — Git ignores `.git/hooks` entirely once the path is redirected.

## What They Check

`commit-msg` requires a `<Verb>: <Capitalized summary>` subject with an approved verb, and rejects AI-assistant attribution trailers.

`pre-commit` runs the style checks against __the added lines of each staged file__, then the formatter gates.
A file that already contains a violation stays committable; a diff that introduces one does not.
Each formatter gate is skipped when its tool is absent, so a fresh clone without `clang-format`, `cmake-format`, or `ruff` still commits.

## Bypassing

```sh
git commit --no-verify
```

There is no in-band escape comment, deliberately.
If a rule needs to bend for a legitimate one-off, edit the check here and document why in the comment above it.
