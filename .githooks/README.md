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

## Testing The Checks

```sh
.githooks/selftest
```

Each fixture is a throwaway repository outside this tree, and the real hook decides its fate.
__A rejection counts only when exactly one check fired and it was the expected one__, which turns every negative case into a false-positive test as well: a fixture tripping two rules means some other check is reading text it has no business reading.

Run it after touching any check.
The recurring failure in this file is not a missed violation but a false positive on prose — the house style requires a rationale comment above each rule, and every rationale names the construct its rule bans, so a check that reads raw lines rejects its own documentation.
That is why the code rules read `code_only`, the prose rules read `strip_literals` and keep their own comment anchors, and the few that inspect comment syntax itself read the raw line and say so where they are defined.

## Portability

bash 3.2 and POSIX `grep`/`sed` are enough, so a stock macOS shell and Git for Windows both run these unchanged.
No `mapfile`, no associative arrays, no `${var,,}`, and no `;;&` case fallthrough — that one is bash 4, and on the bash 3.2 macOS still ships it is a parse error that kills the whole script rather than one check.
Diff text is stripped of CR before matching, and `* text=auto eol=lf` in `.gitattributes` is load-bearing: Git for Windows turns `core.autocrlf` on by default, and a CRLF checkout gives these files a shebang ending in a carriage return, at which point every hook exits 127 and nothing runs at all.

## Bypassing

```sh
git commit --no-verify
```

There is no in-band escape comment, deliberately.
If a rule needs to bend for a legitimate one-off, edit the check here and document why in the comment above it.
