# PTMCMCSampler Release Process

**Status**: Manual release process (requires GitHub release + conda feedstock update)

## Prerequisites
- Maintainer access to [nanograv/PTMCMCSampler](https://github.com/nanograv/PTMCMCSampler)
- Maintainer access to [conda-forge/ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock)
- PyPI credentials configured in GitHub secrets (`PYPI_USERNAME`, `PYPI_PASSWORD`)

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

### 5. Update Conda Feedstock (Manual)
**⚠️ IMPORTANT: Wait for PyPI package to be available first!**

1. **Wait for PyPI upload to complete** (usually 5-10 minutes after GitHub release)
   - Check [PyPI package page](https://pypi.org/project/ptmcmcsampler/) for your new version
   - Verify the package files are downloadable

2. **Get SHA256 hash from PyPI**:
   - Go to your package page on PyPI
   - Click on the version number
   - Copy the SHA256 hash from the package details (when trying to download .tar.gz file)

3. **Update conda feedstock**:
   - Go to [ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock) and clone the repository if you do not have it yet
   - Edit `recipe/meta.yaml`:
     - Update version number (line 2: `{% set version = "2.1.3" %}`)
     - Update SHA256 hash with the value from PyPI
     - Commit to a release branch, and push to your personal github repository/branch
   - Submit PR to conda-forge
   - After submitting the PR, add this exact comment to the PR: `@conda-forge-admin, please rerender`
   - Conda-forge bot will handle the rest

4. **Wait for conda package** (usually 1-2 hours after feedstock PR is merged)
   - Check [conda-forge page](https://anaconda.org/conda-forge/ptmcmcsampler) for availability

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
