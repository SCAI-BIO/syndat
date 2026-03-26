import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.run_install(
        "uv",
        "sync",
        "--frozen",
        "--python",
        session.python,
        "--group",
        "dev",
        external=True,
    )
    session.run(
        "uv",
        "run",
        "--frozen",
        "--python",
        session.python,
        "pytest",
        *session.posargs,
        external=True,
    )
