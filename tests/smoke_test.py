def test_package_imports():
    """
    Basic check that the installed wheel and its core submodules can be imported.
    This helps catch missing third-party dependencies in the build configuration.
    """
    import syndat

    assert syndat is not None

    from syndat import scores, metrics, visualization, postprocessing, rct

    assert all([scores, metrics, visualization, postprocessing, rct])


def test_version_exists():
    """
    Ensure version metadata is accessible.
    """
    from syndat import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0