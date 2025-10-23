# PTMCMCSampler Release Process

**Status**: Manual release process (requires GitHub release + conda feedstock update)

## Prerequisites
- Maintainer access to [nanograv/PTMCMCSampler](https://github.com/nanograv/PTMCMCSampler)
- Maintainer access to [conda-forge/ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock)
- PyPI credentials configured in GitHub secrets (`PYPI_USERNAME`, `PYPI_PASSWORD`). Current maintainers know about this, but should be set up properly already.

## Release Steps

### 1. Prepare Release
```bash
# Ensure main branch is up to date
git checkout master
git pull origin master

# Run tests locally
make test

# Check that all CI tests are passing on GitHub
# Visit: https://github.com/nanograv/PTMCMCSampler/actions
```

### 2. Create Git Tag
```bash
# Create and push the git tag
git tag v2.1.3
git push origin v2.1.3
```

### 3. Create GitHub Release
1. Go to [PTMCMCSampler releases](https://github.com/nanograv/PTMCMCSampler/releases)
2. Click "Create a new release"
3. Choose the tag version you just created (e.g., `v2.1.3`)
4. Set release title (e.g., `v2.1.3`)
5. Add release notes describing changes
6. Click "Publish release"

### 4. Automated PyPI Upload
- GitHub Actions will automatically:
  - Run tests across Python 3.8-3.11 and Ubuntu/macOS
  - Build source distribution and wheel
  - Test the built packages (`make test-sdist`, `make test-wheel`)
  - Upload to PyPI using stored credentials

### 5. Update Conda Feedstock (Semi-Automated)
**⚠️ IMPORTANT: Wait for PyPI package to be available first!**

1. **Wait for PyPI upload to complete** (usually 5-10 minutes after GitHub release)
   - Check [PyPI package page](https://pypi.org/project/ptmcmcsampler/) for your new version
   - Verify the package files are downloadable

2. **Wait for conda-forge bot PR** (usually 30 minutes after PyPI release)
   - The `regro-cf-autotick-bot` will automatically create a PR
   - Go to [ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock) and look for the bot PR
   - The bot will have already updated version number and SHA256 hash

3. **Modify the bot's PR (if needed)**:
   - Clone the [ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock) repository
   - Check out the bot's branch (usually named like `2.1.4_ha7682d`)
   - Make any necessary fixes to `recipe/meta.yaml` (if needed)
   - Commit and push directly to the bot's branch

4. **Add rerender comment**:
   - Add this exact comment to the PR: `@conda-forge-admin, please rerender`
   - Conda-forge bot will handle the rest

5. **Wait for conda package** (usually 1-2 hours after PR is merged)
   - Check [conda-forge page](https://anaconda.org/conda-forge/ptmcmcsampler) for availability

**Fallback if bot doesn't create PR within 1 hour:**
- Manually update `recipe/meta.yaml` with the new version and SHA256 hash from [PyPI](https://pypi.org/project/ptmcmcsampler/)
- Create a PR to the feedstock as described in the [conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs.html)

**Common Build Issues:**
- **Python version compatibility**: If build fails for new Python versions (e.g., 3.14), add `--no-build-isolation` to the pip install command in `meta.yaml`
- **setuptools errors**: Use `--no-build-isolation` to use conda-forge's setuptools instead of PyPI's

## Timing Considerations

### PyPI Package Availability
- **Upload time**: 5-10 minutes after GitHub release is published
- **Verification**: Check [PyPI package page](https://pypi.org/project/ptmcmcsampler/) for your version
- **SHA256 hash**: Available immediately after upload completes (at the tar.gz download)

### Conda Package Availability  
- **Feedstock PR**: Submit after PyPI package is available
- **Build time**: 1-2 hours after feedstock PR is merged
- **Verification**: Check [conda-forge page](https://anaconda.org/conda-forge/ptmcmcsampler)

## Version Management
- Uses `setuptools_scm` - version automatically generated from git tags
- No manual version file updates needed
- Version is written to `PTMCMCSampler/version.py` automatically

## Troubleshooting
- **PyPI upload fails**: Check GitHub secrets are configured correctly
- **Conda build fails**: Check dependencies in `meta.yaml` are up to date
- **Tests fail**: Ensure all dependencies are installed locally before release

## Release Checklist
- [ ] All tests passing on GitHub Actions
- [ ] Release notes prepared
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] PyPI upload successful (check [PyPI page](https://pypi.org/project/ptmcmcsampler/))
- [ ] Conda feedstock updated
- [ ] Conda package available (check [conda-forge page](https://anaconda.org/conda-forge/ptmcmcsampler))
