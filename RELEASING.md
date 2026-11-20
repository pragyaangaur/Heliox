# Releasing Heliox

This page covers the one-time setup for publishing Heliox to PyPI, and the steps for every release after that. The release workflow is `.github/workflows/release.yml`. It builds the wheels and the source distribution, tests each wheel in a clean environment, and attaches the files to the GitHub release. It publishes to PyPI only when the setup below is complete.

## One-time setup on PyPI

These steps need your accounts, so only you can do them.

1. Sign in to pypi.org and turn on two-factor authentication for your account.
2. Open your account settings, go to Publishing, and choose Add a new pending publisher. Fill in these fields:
   - PyPI project name: `heliox`
   - Owner: `pragyaangaur`
   - Repository name: `Heliox`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   A pending publisher lets the first upload create the project. No API token is needed. The workflow exchanges a short-lived GitHub token for a PyPI upload token each time it runs.

## One-time setup on GitHub

1. Open the Heliox repository on GitHub and go to Settings, then Environments. Create an environment named `pypi`. If you want a pause before each upload, add yourself under Required reviewers.
2. Go to Settings, then Secrets and variables, then Actions, then the Variables tab. Add a repository variable named `PUBLISH_TO_PYPI` with the value `true`.

The variable is the switch. While it is missing or set to anything other than `true`, published releases still get their files attached to the release page, but nothing is uploaded to PyPI.

## Checking the build without publishing

Open the Actions tab, choose the Release workflow, and click Run workflow. This builds every wheel and the source distribution, runs the smoke tests, and uploads the files as workflow artifacts. It never publishes, even when `PUBLISH_TO_PYPI` is `true`, because publishing only happens on a published release.

## Publishing the first release

1. Merge the release branch into `main`.
2. Run the check above and make sure every job is green.
3. Go to the Releases page on GitHub and choose Draft a new release. Create a tag named `v0.2.0` on `main`, title it `Heliox 0.2.0`, paste the 0.2.0 section of `CHANGELOG.md` into the notes, and click Publish release.
4. The workflow runs again on the published release. It attaches the wheels and the source distribution to the release, then publishes them to PyPI if `PUBLISH_TO_PYPI` is `true`.
5. Check https://pypi.org/project/heliox/ and install the package in a new environment with `pip install heliox==0.2.0`.

If the publish step fails before any file reaches PyPI, fix the problem and re-run the failed jobs from the workflow run page. Publishing a file to PyPI cannot be undone, and PyPI never accepts the same version number twice. If a file is uploaded with a mistake, release a new version instead.

## Cutting a future release

1. Pick the new version number using semantic versioning.
2. Change the version in four places: `version` in `pyproject.toml`, `version` in `core/Cargo.toml`, the `heliox-core` entry in `core/Cargo.lock`, and `__version__` in `heliox/version.py`. The smoke test fails if the wheel version and `heliox.__version__` do not match.
3. In `CHANGELOG.md`, rename the `[Unreleased]` heading to the new version and date, and add an empty `[Unreleased]` section above it.
4. Merge the change into `main`, then run the check described above.
5. Publish a GitHub release with a tag named after the new version, such as `v0.3.0`.

The version in the tag and the version in the files should match. The workflow does not check the tag, so compare them yourself before publishing.
